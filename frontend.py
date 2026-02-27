import streamlit as st
import requests

# 🔴 REPLACE THIS WITH YOUR RENDER BACKEND URL
BACKEND_URL = "https://your-backend-name.onrender.com/recommend"

st.set_page_config(
    page_title="Alternate Product Recommendation System using AI",
    page_icon="🛍️",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = 1

if "results" not in st.session_state:
    st.session_state.results = []


@st.cache_data(show_spinner=False)
def fetch_results(query):
    try:
        response = requests.post(
            BACKEND_URL,
            json={"query": query, "n": 100},
            timeout=20
        )
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        return []


st.title("🛍️ Alternate Product Recommendation System using AI")

query = st.text_input("Search for a product")

if st.button("🔍 Search"):

    if query.strip() == "":
        st.warning("Please enter something to search.")
    else:
        with st.spinner("Finding best matches..."):
            st.session_state.results = fetch_results(query)
            st.session_state.page = 1


results = st.session_state.results

if results:

    items_per_page = 12
    total_pages = max(1, (len(results) - 1) // items_per_page + 1)

    start = (st.session_state.page - 1) * items_per_page
    end = start + items_per_page
    page_items = results[start:end]

    st.subheader("🔥 Recommended Products")

    cols_per_row = 3

    for i in range(0, len(page_items), cols_per_row):
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            if i + j < len(page_items):
                product = page_items[i + j]

                with cols[j]:
                    if product.get("image_url"):
                        st.image(product["image_url"], use_container_width=True)

                    st.markdown(f"**{product.get('title','No Title')}**")
                    st.markdown(f"₹ {product.get('price','N/A')}")
                    st.markdown(f"⭐ {product.get('rating','N/A')}")
                    st.markdown(f"Platform: {product.get('platform','')}")

                    if product.get("product_url"):
                        st.markdown(f"[🛒 View Product]({product['product_url']})")

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if st.button("⬅ Previous") and st.session_state.page > 1:
            st.session_state.page -= 1

    with col3:
        if st.button("Next ➡") and st.session_state.page < total_pages:
            st.session_state.page += 1

    st.write(f"Page {st.session_state.page} of {total_pages}")
