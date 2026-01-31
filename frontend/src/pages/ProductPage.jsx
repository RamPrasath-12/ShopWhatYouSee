
import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const ProductPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { item, movie, llmFilters, scene } = location.state || {}; // Enhanced data from Watch page

    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);

    // Dynamic mock details based on detected class (fallback)
    const category = item ? item.class.toLowerCase() : "unknown";
    const attributes = item?.attributes; // AG-MAN attributes (if available)

    const mockTitles = {
        "shirt": "Men's Classic Regular Fit Cotton Formal Shirt",
        "tshirt": "Premium Cotton Crew Neck T-Shirt - Urban Style",
        "jacket": "Vintage Denim Trucker Jacket with Sherpa Lining",
        "dress": "Women's Elegant A-Line Evening Dress",
        "pants": "Slim Fit Chinos | Stretchable Fabric",
        "cap": "Sports Baseball Cap - Adjustable Strap"
    };

    const mockPrice = {
        "shirt": 499,
        "tshirt": 299,
        "jacket": 1299,
        "dress": 899,
        "pants": 699,
        "cap": 199
    };

    useEffect(() => {
        if (item) {
            setLoading(true);

            // Use LLM-generated filters if available, otherwise fallback to basic filters
            const searchFilters = llmFilters || {
                category: item.class,
                // Use AG-MAN attributes as fallback
                color: attributes?.color_hex,
                pattern: attributes?.pattern,
                sleeve_length: attributes?.sleeve_length
            };

            console.log("🔍 Product search with filters:", searchFilters);
            console.log("📊 AG-MAN Attributes:", attributes);
            console.log("🎬 Scene context:", scene);

            axios.post('http://localhost:5000/search', {
                detected_category: item.class,
                filters: searchFilters
            })
                .then(res => {
                    console.log("✅ Search results:", res.data);
                    if (res.data.products && res.data.products.length > 0) {
                        setProducts(res.data.products);
                    }
                    setLoading(false);
                })
                .catch(err => {
                    console.error("❌ Backend search failed, using mock data:", err);
                    setLoading(false);
                });
        }
    }, [item]);

    if (!item) {
        return <div style={{ color: 'white', padding: 20 }}>No product selected.</div>;
    }

    const mainProduct = products.length > 0 ? products[0] : null;

    // Use backend data if available, otherwise fallback to mock
    const title = mainProduct ? mainProduct.name : (mockTitles[category] || `Premium Quality ${item.class}`);
    const price = mainProduct ? mainProduct.price : (mockPrice[category] || 499);

    return (
        <div style={styles.container}>
            {/* Header */}
            <header style={styles.header}>
                <div style={styles.logo} onClick={() => navigate('/')}>amazon.in</div>
                <div style={styles.searchBar}>
                    <input type="text" value={title} readOnly style={styles.searchInput} />
                    <button style={styles.searchButton}>🔍</button>
                </div>
                <div style={styles.cart}>🛒 Cart</div>
            </header>

            {/* Back to Movie */}
            <div style={styles.backBar} onClick={() => navigate(-1)}>
                ← Back to {movie ? movie.title : 'Movie'}
            </div>

            {/* Product Grid */}
            <div style={styles.productSection}>
                {/* Visual Analysis Image (The Detection) */}
                <div style={styles.imageColumn}>
                    <div style={styles.imageContainer}>
                        <img
                            src={`data:image/jpeg;base64,${item.cropped_image}`}
                            alt={item.class}
                            style={styles.productImage}
                        />
                        <div style={styles.tag}>IDENTIFIED IN VIDEO</div>
                    </div>
                    {/* Show backend result image if available */}
                    {mainProduct && (
                        <div style={{ marginTop: '20px', textAlign: 'center' }}>
                            <p style={{ color: '#555', fontSize: '12px', fontWeight: 'bold' }}>MATCHED PRODUCT</p>
                            <img
                                src={mainProduct.image_url}
                                alt={mainProduct.name}
                                style={{
                                    width: '150px',
                                    height: '150px',
                                    objectFit: 'contain',
                                    border: '1px solid #ddd',
                                    borderRadius: '4px'
                                }}
                            />
                        </div>
                    )}
                </div>

                {/* Details Column */}
                <div style={styles.detailsColumn}>
                    <h1 style={styles.productTitle}>{title}</h1>
                    <div style={styles.rating}>⭐⭐⭐⭐☆ <span style={{ color: '#007185', fontSize: '14px' }}>1,240 ratings</span></div>

                    <div style={styles.priceBlock}>
                        <span style={styles.currency}>₹</span>
                        <span style={styles.price}>{price}</span>
                        <span style={styles.decimals}>00</span>
                    </div>

                    <div style={styles.description}>
                        • Material: 100% Premium Quality Fabric<br />
                        • Fit: Regular Fit<br />
                        • Occasion: Casual & Formal Wear<br />
                        • Care Instructions: Machine Wash<br />
                        • <b>Shop the Look:</b> This exact item was identified using AI from the scene you were watching.
                    </div>

                    <div style={styles.badges}>
                        <span style={styles.badge}>✔ Prime</span>
                        <span style={styles.delivery}>FREE Delivery by Tomorrow, 2 PM</span>
                    </div>

                    {/* Similar Items from Backend */}
                    {products.length > 1 && (
                        <div style={styles.similarSection}>
                            <h3 style={styles.similarTitle}>Similar items from our store</h3>
                            <div style={styles.similarGrid}>
                                {products.slice(1).map(prod => (
                                    <div key={prod.id} style={styles.similarCard}>
                                        <img src={prod.image_url} alt={prod.name} style={styles.similarImage} />
                                        <div style={styles.similarName}>{prod.name}</div>
                                        <div style={styles.similarPrice}>₹{prod.price}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Buy Box */}
                <div style={styles.buyBox}>
                    <div style={styles.priceBlockSmall}>
                        <sup style={{ fontSize: '12px' }}>₹</sup>{price}<sup style={{ fontSize: '12px' }}>00</sup>
                    </div>
                    <div style={{ color: '#007600', marginBottom: '15px', fontWeight: '500' }}>In stock</div>

                    <button style={styles.addToCart}>Add to Cart</button>
                    <button style={styles.buyNow}>Buy Now</button>
                </div>
            </div>
        </div>
    );
};

const styles = {
    container: {
        backgroundColor: "white",
        minHeight: "100vh",
        fontFamily: "Arial, sans-serif"
    },
    header: {
        background: "#131921",
        color: "white",
        padding: "10px 20px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between"
    },
    logo: {
        fontSize: "24px",
        fontWeight: "bold",
        cursor: "pointer",
        fontStyle: "italic"
    },
    searchBar: {
        flex: 1,
        margin: "0 40px",
        display: "flex"
    },
    searchInput: {
        width: "100%",
        padding: "10px",
        borderRadius: "4px 0 0 4px",
        border: "none",
        fontSize: "16px"
    },
    searchButton: {
        padding: "0 15px",
        background: "#febd69",
        border: "none",
        borderRadius: "0 4px 4px 0",
        cursor: "pointer",
        fontSize: "20px"
    },
    cart: {
        fontWeight: "bold",
        cursor: "pointer"
    },
    backBar: {
        background: "#232f3e",
        color: "white",
        padding: "10px 20px",
        fontSize: "14px",
        cursor: "pointer"
    },
    productSection: {
        display: "flex",
        padding: "40px",
        maxWidth: "1400px",
        margin: "0 auto",
        gap: "40px"
    },
    imageColumn: {
        flex: "0 0 35%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center"
    },
    imageContainer: {
        position: 'relative'
    },
    productImage: {
        maxHeight: "500px",
        maxWidth: "100%",
        objectFit: "contain"
    },
    tag: {
        position: "absolute",
        bottom: "20px",
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,0,0,0.8)",
        color: "white",
        padding: "5px 10px",
        borderRadius: "20px",
        fontSize: "12px",
        fontWeight: "bold"
    },
    detailsColumn: {
        flex: 1
    },
    productTitle: {
        fontSize: "28px",
        lineHeight: "1.3",
        fontWeight: "normal",
        marginBottom: "10px"
    },
    rating: {
        color: "#ffa41c",
        marginBottom: "15px"
    },
    priceBlock: {
        display: "flex",
        alignItems: "flex-start",
        marginBottom: "15px"
    },
    currency: {
        fontSize: "14px",
        paddingTop: "5px"
    },
    price: {
        fontSize: "28px",
        fontWeight: "500",
        lineHeight: "1"
    },
    decimals: {
        fontSize: "14px",
        paddingTop: "5px"
    },
    description: {
        fontSize: "16px",
        lineHeight: "1.6",
        color: "#333",
        marginTop: "20px"
    },
    badges: {
        marginTop: "20px",
        display: "flex",
        flexDirection: "column",
        gap: "5px"
    },
    badge: {
        color: "#00a8e1",
        fontWeight: "bold",
        fontSize: "14px"
    },
    delivery: {
        color: "#565959",
        fontSize: "14px"
    },
    buyBox: {
        flex: "0 0 250px",
        border: "1px solid #d5d9d9",
        borderRadius: "8px",
        padding: "20px",
        height: "fit-content"
    },
    priceBlockSmall: {
        color: "#B12704",
        fontSize: "20px",
        fontWeight: "bold",
        marginBottom: "10px"
    },
    addToCart: {
        width: "100%",
        padding: "10px",
        borderRadius: "20px",
        border: "none",
        background: "#ffd814",
        marginBottom: "10px",
        cursor: "pointer",
        fontSize: "14px"
    },
    buyNow: {
        width: "100%",
        padding: "10px",
        borderRadius: "20px",
        border: "none",
        background: "#ffa41c",
        cursor: "pointer",
        fontSize: "14px"
    },
    similarSection: {
        marginTop: "40px",
        borderTop: "1px solid #ddd",
        paddingTop: "20px"
    },
    similarTitle: {
        fontSize: "18px",
        marginBottom: "15px",
        color: "#CC6600"
    },
    similarGrid: {
        display: "flex",
        gap: "15px",
        overflowX: "auto",
        paddingBottom: "10px"
    },
    similarCard: {
        border: "1px solid #eee",
        borderRadius: "4px",
        padding: "10px",
        width: "120px",
        textAlign: "center"
    },
    similarImage: {
        width: "100%",
        height: "100px",
        objectFit: "contain",
        marginBottom: "5px"
    },
    similarName: {
        fontSize: "12px",
        height: "30px",
        overflow: "hidden",
        marginBottom: "5px"
    },
    similarPrice: {
        color: "#B12704",
        fontWeight: "bold",
        fontSize: "13px"
    }
};

export default ProductPage;
