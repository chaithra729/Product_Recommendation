import requests

def is_valid_image_url(url):
    try:
        response = requests.head(url, timeout=5)  # Send a HEAD request
        content_type = response.headers.get('Content-Type', '')
        return response.status_code == 200 and content_type.startswith('image')
    except requests.RequestException:
        return False

# Example: Validate all URLs in the trending_products DataFrame
trending_products['ValidImage'] = trending_products['ImageURL'].apply(is_valid_image_url)

# Filter out invalid image URLs
valid_trending_products = trending_products[trending_products['ValidImage']]
