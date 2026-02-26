import streamlit as st
import requests

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
            "http://127.0.0.1:8000/recommend",
            json={"query": query, "n": 100},
            timeout=15
        )
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except:
        return []


st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.stApp {
    background: transparent;
}

/* Title */
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #00e0ff;
    margin-bottom: 25px;
    font-family: 'Trebuchet MS', sans-serif;
    text-shadow: 0 0 20px rgba(0,224,255,0.7);
}

/* Card */
.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(15px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: 0.4s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    color: white;
    animation: fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 40px rgba(0,224,255,0.6);
}

.card h3 {
    font-size: 18px;
    line-height: 1.4em;
    height: 2.8em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.price {
    color: #00ff9d;
    font-weight: bold;
    font-size: 18px;
}

.platform-logo {
    height: 24px;
    margin-top: 8px;
}

/* SEARCH BAR FIX */
.stTextInput>div>div>input {
    background: rgba(255,255,255,0.9) !important;
    color: #000000 !important;
    font-size: 16px !important;
    border-radius: 30px !important;
    border: 1px solid rgba(0,0,0,0.5) !important;
    padding: 12px 20px !important;
}

.stTextInput>div>div>input::placeholder {
    color: #555555 !important;
    opacity: 1 !important;
}

.stButton>button {
    background: linear-gradient(45deg, #00e0ff, #007cf0) !important;
    color: white !important;
    border-radius: 30px;
    border: none;
    padding: 10px 25px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">Alternate Product Recommendation System using AI</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    query = st.text_input(
        "Search",
        placeholder="Search for a product...",
        label_visibility="collapsed"
    )
    search_clicked = st.button("🔍 Search")

st.markdown("### 🔎 Filters")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    sort_option = st.selectbox(
        "Sort By",
        ["Relevance", "Price Low to High", "Price High to Low", "Rating High to Low"]
    )

    platform_filter = st.selectbox(
        "Platform",
        ["All", "Amazon", "Flipkart"]
    )

with filter_col2:
    min_price = st.number_input("Minimum Price", min_value=0, value=0)
    max_price = st.number_input("Maximum Price", min_value=0, value=100000)

st.markdown("---")

if search_clicked:
    st.session_state.page = 1

    if query.strip() == "":
        st.warning("Please enter something to search.")
    else:
        with st.spinner("Finding best matches..."):
            st.session_state.results = fetch_results(query)

results = st.session_state.results

if results:

    filtered = []

    for r in results:
        try:
            price = float(r.get("price", 0))
        except:
            price = 0

        if min_price <= price <= max_price:
            if platform_filter == "All" or r.get("platform") == platform_filter:
                filtered.append(r)

    if sort_option == "Price Low to High":
        filtered = sorted(filtered, key=lambda x: float(x.get("price", 0)))
    elif sort_option == "Price High to Low":
        filtered = sorted(filtered, key=lambda x: float(x.get("price", 0)), reverse=True)
    elif sort_option == "Rating High to Low":
        filtered = sorted(filtered, key=lambda x: float(x.get("rating", 0)), reverse=True)

    items_per_page = 16
    total_pages = max(1, (len(filtered) - 1) // items_per_page + 1)

    start = (st.session_state.page - 1) * items_per_page
    end = start + items_per_page
    page_items = filtered[start:end]

    st.markdown("## 🔥 Recommended Products")

    columns_count = 4

    for row in range(0, len(page_items), columns_count):
        cols = st.columns(columns_count)
        for i in range(columns_count):
            if row + i < len(page_items):
                product = page_items[row + i]
                with cols[i]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    if product.get("image_url"):
                        st.markdown(
                            f"""
                            <div style="width:100%; aspect-ratio:1/1; overflow:hidden; border-radius:12px;">
                                <img src="{product['image_url']}" 
                                     style="width:100%; height:100%; object-fit:cover;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"<h3>{product.get('title','No Title')}</h3>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"<div class='price'>₹ {product.get('price','N/A')}</div>",
                        unsafe_allow_html=True
                    )

                    platform = product.get("platform", "").lower()
                    if "amazon" in platform:
                        st.markdown(
                            '<img class="platform-logo" src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg">',
                            unsafe_allow_html=True
                        )
                    elif "flipkart" in platform:
                        st.markdown(
                            '<img class="platform-logo" src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Flipkart_logo.svg" style="background:white; padding:4px; border-radius:6px;">',
                            unsafe_allow_html=True
                        )

                    if product.get("product_url"):
                        st.markdown(f"[🛒 View Product]({product['product_url']})")

                    st.markdown('</div>', unsafe_allow_html=True)

    
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if st.button("⬅ Previous") and st.session_state.page > 1:
            st.session_state.page -= 1

    with col3:
        if st.button("Next ➡") and st.session_state.page < total_pages:
            st.session_state.page += 1

    st.markdown(
        f"<center style='color:white;'>Page {st.session_state.page} of {total_pages}</center>",
        unsafe_allow_html=True
    )