from flask import Flask, request, render_template, redirect, session, flash, url_for, make_response
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from flask_migrate import Migrate

import os

app = Flask(__name__, instance_relative_config=True)
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"

os.makedirs(app.instance_path, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'ecom.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)


# --------------------- MODELS ---------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    interactions = db.relationship('Interaction', backref='user', lazy=True)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_name = db.Column(db.String(255), nullable=False)
    interaction_type = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())

class CartItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    product_name = db.Column(db.String(100))
    quantity = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('cart_items', lazy=True))

class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    product_name = db.Column(db.String(100), nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('purchases', lazy=True))

# ------------------ HELPERS -------------------
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text

# ------------------ DATA LOAD -------------------
trending_products = pd.read_csv("product_images_100000.csv")
trending_products.columns = trending_products.columns.str.strip()

if 'Product Name' in trending_products.columns:
    trending_products.rename(columns={'Product Name': 'Name'}, inplace=True)

train_data = pd.read_csv("data/filtered_data.csv")
train_data.columns = train_data.columns.str.strip()

column_rename_map = {
    'Product Name': 'Name',
    'Product Brand': 'Brand',
    'Product Reviews Count': 'ReviewCount',
    'Product Rating': 'Rating',
    'Product Price': 'Price'
}
train_data.rename(columns=column_rename_map, inplace=True)

required_cols = ['Name', 'Brand', 'ReviewCount', 'Rating', 'Price']
for col in required_cols:
    if col not in train_data.columns:
        train_data[col] = 'N/A'

# ------------------ CONTENT-BASED FILTERING -------------------
def content_based_recommendations(query, top_n=10):
    df = train_data.copy()
    df['combined'] = df['Name'].fillna('') + " " + df['Brand'].fillna('') + " " + df['Rating'].astype(str)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])

    product_names = df['Name'].fillna('').unique()
    matched_query, score = process.extractOne(query, product_names)

    if score < 80:
        return pd.DataFrame()

    matched_indices = df[df['Name'] == matched_query].index
    if matched_indices.empty:
        return pd.DataFrame()

    idx = matched_indices[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    df['similarity'] = cosine_similarities

    # 🏎️ Vectorized filter: Name contains query
    filtered_df = df[df['Name'].str.lower().str.contains(query.lower())].copy() # Create a copy to avoid SettingWithCopyWarning

    # Sort by similarity (descending) and then by Rating (descending)
    filtered_df = filtered_df.sort_values(by=['similarity', 'Rating'], ascending=[False, False]).head(top_n)

    return filtered_df[['Name', 'Brand', 'ReviewCount', 'Rating', 'Price']]




# ------------------ COLLABORATIVE UTILS -------------------
def create_user_item_matrix():
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    try:
        interactions_df = pd.read_sql_table('interaction', con=engine)
    except:
        return pd.DataFrame()
    if interactions_df.empty:
        return pd.DataFrame()
    return interactions_df.pivot_table(index='user_id', columns='product_name', values='id', aggfunc='count').fillna(0)

def train_knn_model(user_item_matrix, n_neighbors=5):
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model.fit(user_item_matrix)
    return model

def get_knn_recommendations(user_id, user_item_matrix, model, top_n=5):
    if user_id not in user_item_matrix.index:
        return [], []  # Return empty lists if user not found
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    n_neighbors_to_find = min(top_n + 1, user_item_matrix.shape[0])
    distances, indices = model.kneighbors(user_vector, n_neighbors=n_neighbors_to_find)
    neighbor_indices = indices.flatten()[1:]  # Exclude the user itself
    neighbor_ids = user_item_matrix.iloc[neighbor_indices].index.tolist() # Get the actual User IDs of neighbors

    known_products = user_item_matrix.loc[user_id]
    neighbor_products = user_item_matrix.loc[neighbor_ids].sum()
    already_interacted = known_products[known_products > 0].index.tolist()
    recommended_products = neighbor_products[~neighbor_products.index.isin(already_interacted)]
    recommendations = recommended_products.sort_values(ascending=False).head(top_n).index.tolist()

    return recommendations, neighbor_ids


# ------------------ ROUTES -------------------
@app.route("/")
@app.route("/index")
def index():
    top_products = trending_products.head(8).copy()
    top_products.columns = top_products.columns.str.strip()
    train_data.columns = train_data.columns.str.strip()
    top_products = pd.merge(top_products, train_data[required_cols], on='Name', how='left')
    top_products['Brand'] = top_products['Brand'].fillna('Unknown')
    prices = top_products['Price'].fillna('N/A').tolist()
    return render_template('index.html', trending_products=top_products, truncate=truncate, prices=prices)

@app.route("/main")
def main():
    if 'username' not in session:
        flash("Please register or sign in first to access recommendations.")
        return redirect(url_for('index'))

    user = User.query.filter_by(username=session['username']).first()
    collaborative_rec = pd.DataFrame()
    trending_fallback = trending_products.head(5).copy()
    trending_fallback = pd.merge(trending_fallback, train_data[required_cols], on='Name', how='left')
    trending_fallback['Brand'] = trending_fallback['Brand'].fillna('Unknown')
    prices = trending_fallback['Price'].fillna('N/A').tolist()
    message = None

    user_item_matrix = create_user_item_matrix()
    

    print("Index:", user_item_matrix.index.tolist())
   

    if not user_item_matrix.empty and user.id in user_item_matrix.index:
        print(f"User ID: {user.id}")
      
        model = train_knn_model(user_item_matrix)
        recommended_names, neighbor_ids = get_knn_recommendations(user.id, user_item_matrix, model, top_n=5)
        print("Recommended Names:", recommended_names)
        print("Neighbor IDs for User {}: {}".format(user.id, neighbor_ids))  # Keep this line

        if recommended_names:
            collaborative_rec = train_data[train_data['Name'].isin(recommended_names)].copy()

            # --- New Code to Display Neighbor Interactions ---
            neighbor_interactions = user_item_matrix.loc[neighbor_ids, recommended_names]
            print("\n--- Neighbor Interactions ---")
            print(neighbor_interactions)
            # --- End of New Code ---
            # --- New Code to Display Common Interactions ---
            target_user_interactions = user_item_matrix.loc[user.id]
            common_interactions = target_user_interactions[target_user_interactions > 0].index.tolist()
            neighbor_interactions_all = user_item_matrix.loc[neighbor_ids]
            common_interactions_with_neighbors = {}

            for neighbor_id in neighbor_ids:
                neighbor_interaction_products = neighbor_interactions_all.loc[neighbor_id]
                neighbor_interactions_products_list = neighbor_interaction_products[neighbor_interaction_products > 0].index.tolist()
                common_products = list(set(common_interactions) & set(neighbor_interactions_products_list))
                common_interactions_with_neighbors[neighbor_id] = common_products

            print(f"\n--- Common Interactions with User {user.id} ---")
            for neighbor_id, common_products in common_interactions_with_neighbors.items():
                print(f"Neighbor {neighbor_id}: {common_products}")
# --- End of New Code ---
            if not collaborative_rec.empty:
                collaborative_rec.drop_duplicates(subset=['Name'], keep='first', inplace=True)
                collaborative_rec = collaborative_rec.sort_values(by='Rating', ascending=False)
        else:
            message = "Based on current interactions, we don't have personalized recommendations yet. Please interact with more products."
    else:
        message = "Not enough user interaction data to generate collaborative recommendations yet. Please browse and interact with products."

    return render_template(
        'main.html',
        content_based_rec=pd.DataFrame(),
        collaborative_rec=collaborative_rec,
        trending_fallback=trending_fallback if collaborative_rec.empty and message else pd.DataFrame(),
        truncate=truncate,
        prices=prices if collaborative_rec.empty and message else [],
        message=message if message and collaborative_rec.empty else None
    )


@app.route("/register", methods=['POST', 'GET'])
def register():
    top_products = trending_products.head(8).copy()
    top_products = pd.merge(top_products, train_data[required_cols], on='Name', how='left')
    prices = top_products['Price'].fillna('N/A').tolist()
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash("Username or email already exists.")
            return render_template('index.html', trending_products=top_products, prices=prices, truncate=truncate)
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        flash("Registration successful!")
        return redirect(url_for('main'))
    return render_template('index.html', trending_products=top_products, prices=prices, truncate=truncate)

@app.route('/login', methods=['POST', 'GET'])
def login():
    top_products = trending_products.head(8).copy()
    top_products = pd.merge(top_products, train_data[required_cols], on='Name', how='left')
    prices = top_products['Price'].fillna('N/A').tolist()
    if request.method == 'POST':
        username = request.form['loginUsername'].strip()
        password = request.form['loginPassword'].strip()
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = username
            flash("Login successful!")
            return redirect(url_for('main'))
        else:
            flash("Invalid username or password.")
            return render_template('index.html', trending_products=top_products, prices=prices, truncate=truncate)
    return render_template('index.html', trending_products=top_products, prices=prices, truncate=truncate)

@app.route("/add_to_cart", methods=['POST'])
def add_to_cart():
    if 'username' not in session:
        flash("Please log in to add items to your cart.")
        return redirect(url_for('index'))
    
    product_name = request.form.get('product_name')
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash("User not found.")
        return redirect(url_for('index'))
    
    existing_item = CartItem.query.filter_by(user_id=user.id, product_name=product_name).first()
    
    if existing_item:
        if existing_item.quantity is None:  # Ensure quantity is initialized
            existing_item.quantity = 0
        existing_item.quantity += 1
    else:
        new_cart_item = CartItem(user_id=user.id, product_name=product_name, quantity=1)
        db.session.add(new_cart_item)
    
    db.session.commit()
    flash(f"Added '{product_name}' to cart!")
    return redirect(url_for('main'))


@app.route("/buy_product", methods=['POST'])
def buy_product():
    if 'username' not in session:
        flash("Please log in to buy products.")
        return redirect(url_for('index'))
    product_name = request.form.get('product_name')
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash("User not found.")
        return redirect(url_for('index'))
    new_purchase = Purchase(user_id=user.id, product_name=product_name)
    db.session.add(new_purchase)
    db.session.commit()
    flash(f"'{product_name}' booked successfully!")
    return redirect(url_for('main'))

@app.route("/purchases")
def view_purchases():
    purchases = Purchase.query.all()
    return "<br>".join([f"{p.user.username} bought {p.product_name} on {p.purchase_date}" for p in purchases])

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You’ve been logged out.")
    return redirect(url_for('index'))


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if 'username' not in session:
        flash("Please register or sign in first to access recommendations.")
        return redirect(url_for('index'))

    top_products = trending_products.head(8).copy()
    top_products = pd.merge(top_products, train_data[required_cols], on='Name', how='left')
    prices = top_products['Price'].fillna('N/A').tolist()

    content_based_rec = pd.DataFrame()
    collaborative_rec = pd.DataFrame()
    message = None

    if request.method == 'POST':
        query = request.form.get('prod')
        nbr_str = request.form.get('nbr')
        nbr = int(nbr_str) if nbr_str and nbr_str.isdigit() else 5

        # ---- Content-Based Filtering ----
        content_based_rec = content_based_recommendations(query, top_n=nbr)
        print("Content-Based Recommendations:")
        print(content_based_rec)
        if content_based_rec.empty:
            message = "No content-based recommendations available."

        user = User.query.filter_by(username=session['username']).first()
        if user:
            # Log interaction
            if not content_based_rec.empty:
                for product_name in content_based_rec['Name'].tolist():
                    interaction = Interaction(user_id=user.id, product_name=product_name, interaction_type='view')
                    db.session.add(interaction)
                db.session.commit()

            # ---- Collaborative Filtering ----
            user_item_matrix = create_user_item_matrix()
            print("--- User-Item Matrix (from /recommendations) ---")
            print(user_item_matrix)
            print("Shape:", user_item_matrix.shape)
            print("Index:", user_item_matrix.index.tolist())
            print("Columns:", user_item_matrix.columns.tolist())
            if not user_item_matrix.empty and user.id in user_item_matrix.index:
                model = train_knn_model(user_item_matrix)
                recommended_names = get_knn_recommendations(user.id, user_item_matrix, model, top_n=nbr)
                if recommended_names:
                    collaborative_rec = train_data[train_data['Name'].isin(recommended_names)].copy()
                    # Sort collaborative recommendations by Rating (descending)
                    collaborative_rec = collaborative_rec.sort_values(by='Rating', ascending=False)
                else:
                    if not message:
                        message = "Based on current interactions, we don't have personalized collaborative recommendations yet."
            else:
                if not message:
                    message = "Not enough user interaction data to generate collaborative recommendations yet."

    return render_template(
        'main.html',
        content_based_rec=content_based_rec,
        collaborative_rec=collaborative_rec,
        truncate=truncate,
        prices=prices,
        message=message
    )

@app.route("/simulate_interactions")
def simulate_interactions():
    if 'username' not in session:
        return redirect(url_for('index'))

    user = User.query.filter_by(username=session['username']).first()
    if user:
        sample_products = train_data['Name'].dropna().sample(5).tolist()
        for name in sample_products:
            db.session.add(Interaction(user_id=user.id, product_name=name, interaction_type='view'))
        db.session.commit()
        flash("Sample interactions added.")
    return redirect(url_for('main'))

    

@app.route("/init_db")
def init_db():
    with app.app_context():
        db.drop_all()
        db.create_all()
    flash("Database initialized!")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
