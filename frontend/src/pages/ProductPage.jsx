
import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const ProductPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { item, movie, llmFilters, llmPriceMax, scene, embedding, sessionHistory } = location.state || {};

    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);

    // ────────────────────────────────────────────────────────
    // 3-LAYER FILTER ARCHITECTURE
    // ────────────────────────────────────────────────────────
    // Layer 1: Visual Baseline (IMMUTABLE after init)
    //   What the camera detected via AGMAN. Never mutated.
    //   Sent to backend as soft scoring signals only.
    const [visualBaseline, setVisualBaseline] = useState(null);

    // Layer 2: User Overrides (only what user explicitly changed)
    //   Each value tagged: { value: "red", source: "user" }
    //   price_max is stored here too: { value: 500, source: "user" }
    //   Sent to backend as hard SQL WHERE constraints.
    const [userOverrides, setUserOverrides] = useState({});

    // Layer 3: Extraction Quality (from AGMAN)
    //   Influences preserved weight in scoring.
    const [extractionQuality, setExtractionQuality] = useState(1.0);

    // Search generation counter — prevents stale results
    const searchGenRef = useRef(0);
    // Prevent double-firing of handleQuerySubmit
    const queryInFlightRef = useRef(false);
    // Prevent double-firing of initial useEffect search
    const initialSearchDoneRef = useRef(false);

    // Product detail modal state
    const [selectedProduct, setSelectedProduct] = useState(null);


    // Dynamic mock details based on detected class
    const category = item ? item.class.toLowerCase() : "unknown";
    const attributes = item?.attributes;

    // Track which product explanations are expanded
    const [expandedExpl, setExpandedExpl] = useState({});

    // ── Analytics: fire tracking events to backend ──
    const sessionId = React.useMemo(() => {
        let sid = sessionStorage.getItem('swys_session');
        if (!sid) { sid = crypto.randomUUID(); sessionStorage.setItem('swys_session', sid); }
        return sid;
    }, []);
    const trackEvent = (eventType, extra = {}) => {
        axios.post('http://localhost:5000/track-event', {
            event_type: eventType,
            session_id: sessionId,
            category: category,
            ...extra,
        }).catch(() => { });  // fire-and-forget
    };

    // ── Canonical key mapper (Problem 5) ──
    const CANONICAL_KEYS = {
        color_family: "color", primary_color_name: "color",
        sleeve_value: "sleeve", sleeve_length: "sleeve",
        pattern_value: "pattern",
        price_bucket: "price", price_max: "price",
    };
    const canonicalize = (key) => CANONICAL_KEYS[key] || key;

    // ── Friendly display labels for filter chips ──
    const DISPLAY_LABELS = {
        color: "Color", sleeve: "Sleeve", pattern: "Pattern",
        category: "Category", gender: "Gender", style: "Style",
        material: "Material", price: "Price",
    };
    const chipLabel = (key) => DISPLAY_LABELS[key] || key;

    // ── Flatten overrides: strip source metadata before sending ──
    // Also extracts price_max separately (backend expects it at top level)
    const flattenOverrides = (overrides) => {
        const flat = {};
        let priceMax = null;
        for (const [k, meta] of Object.entries(overrides)) {
            if (k === 'price') {
                priceMax = meta.value; // price_max sent as top-level param
            } else {
                flat[k] = meta.value;
            }
        }
        return { flat, priceMax };
    };

    // Initialize visual baseline ONCE on first load (immutable)
    useEffect(() => {
        if (item && item.attributes && !visualBaseline) {
            const baseline = {
                category: item.class,
                color_name: item.attributes.color_name || "",
                color_hex: item.attributes.color_hex || "",
                pattern: item.attributes.pattern || "",
                sleeve: item.attributes.sleeve || "",
                gender: llmFilters?.gender || "",
            };
            setVisualBaseline(baseline);
            setExtractionQuality(item.attributes.extraction_quality || 1.0);

            // ── Apply LLM filters from Watch.jsx as user overrides ──
            // This ensures filters like color, sleeve, pattern requested on
            // the Watch page are actually used in the initial search.
            const initialOverrides = {};
            if (llmFilters && typeof llmFilters === 'object') {
                for (const [rawKey, val] of Object.entries(llmFilters)) {
                    if (!val || val === '') continue; // don't skip category so it correctly applies to userOverrides
                    const cKey = canonicalize(rawKey);
                    if (cKey === 'price') continue; // price handled via llmPriceMax
                    // Use the most specific value (e.g. primary_color_name "Red" over color_family "red")
                    if (!initialOverrides[cKey] || rawKey === 'primary_color_name' || rawKey === 'sleeve_value' || rawKey === 'pattern_value') {
                        initialOverrides[cKey] = { value: val, source: 'user' };
                    }
                }
            }
            if (llmPriceMax) {
                initialOverrides.price = { value: llmPriceMax, source: 'user' };
            }
            if (Object.keys(initialOverrides).length > 0) {
                setUserOverrides(prev => ({ ...prev, ...initialOverrides }));
                console.log("[ProductPage] 📋 Watch.jsx LLM filters applied as overrides:", initialOverrides);
            }

            console.log("[ProductPage] 🔒 Visual baseline set (immutable):", baseline);
        }
    }, [item]);

    // Scene label for display
    const sceneLabel = scene?.scene_label
        ? scene.scene_label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
        : null;

    // ── Initial Search: Uses LLM filters from Watch.jsx if present ──
    // Guard: only fire ONCE (React strict mode protection)
    useEffect(() => {
        if (item && !initialSearchDoneRef.current) {
            initialSearchDoneRef.current = true;
            setLoading(true);

            // Baseline attributes from AG-MAN detection
            const baseline = {
                category: item.class,
                color_name: attributes?.color_name || "",
                color_hex: attributes?.color_hex || "",
                sleeve: attributes?.sleeve || "",
                pattern: attributes?.pattern || "",
                gender: llmFilters?.gender || "",
            };

            // ── Build initial user_overrides from Watch.jsx LLM filters ──
            const initOverrides = { category: llmFilters?.category || item.class };
            if (llmFilters && typeof llmFilters === 'object') {
                for (const [rawKey, val] of Object.entries(llmFilters)) {
                    if (!val || val === '' || rawKey === 'category') continue;
                    const cKey = canonicalize(rawKey);
                    if (cKey === 'price') continue;
                    // For duplicates (color_family + primary_color_name both → "color"),
                    // prefer the more specific key
                    if (!initOverrides[cKey] || rawKey === 'primary_color_name' || rawKey === 'sleeve_value' || rawKey === 'pattern_value') {
                        initOverrides[cKey] = val;
                    }
                }
            }

            const gen = ++searchGenRef.current;
            const hasLlmOverrides = Object.keys(initOverrides).length > 1;
            console.log(`[ProductPage] Initial search gen=${gen}${hasLlmOverrides ? ' — WITH Watch.jsx LLM overrides' : ' — PURE_SIMILARITY'}`, { baseline, initOverrides });

            axios.post('http://localhost:5000/search', {
                detected_category: item.class,
                embedding: embedding,
                visual_baseline: baseline,
                user_overrides: initOverrides,
                extraction_quality: attributes?.extraction_quality || 1.0,
                price_max: llmPriceMax || null,
                scene: scene?.scene_label || null,
            })
                .then(res => {
                    if (searchGenRef.current !== gen) {
                        console.log(`[ProductPage] ❌ Ignoring stale initial search gen=${gen}`);
                        return;
                    }
                    if (res.data.products && res.data.products.length > 0) {
                        setProducts(res.data.products);
                    }
                    if (res.data.metadata) {
                        console.log("[ProductPage] V2 Metadata:", res.data.metadata);
                    }
                    setLoading(false);
                })
                .catch(err => {
                    console.error("❌ Search failed:", err);
                    if (searchGenRef.current === gen) setLoading(false);
                });
        }
    }, [item]);

    // Handle User Query (Iterative Refinement)
    const [userQuery, setUserQuery] = useState("");
    const [queryLoading, setQueryLoading] = useState(false);

    // ── Helper: fire search with current state ──
    const fireSearch = async (overrides, gen) => {
        const { flat, priceMax } = flattenOverrides(overrides);
        console.log(`[ProductPage] 🚀 Sending search gen=${gen}`, {
            visual_baseline: visualBaseline,
            user_overrides: flat,
            price_max: priceMax,
            scene: scene?.scene_label,
        });

        const searchRes = await axios.post('http://localhost:5000/search', {
            detected_category: item.class,
            embedding: embedding,
            visual_baseline: visualBaseline,         // scoring only
            user_overrides: flat,                    // hard SQL
            extraction_quality: extractionQuality,
            price_max: priceMax,                     // hard SQL price cap
            scene: scene?.scene_label || null,       // scene context for retrieval
        });

        if (searchGenRef.current !== gen) {
            console.log(`[ProductPage] ❌ Stale after search — gen=${gen}`);
            return;
        }

        if (searchRes.data.products && searchRes.data.products.length > 0) {
            setProducts(searchRes.data.products);
            console.log(`[ProductPage] ✅ Applied ${searchRes.data.products.length} results gen=${gen}`, {
                metadata: searchRes.data.metadata,
            });
        } else {
            console.log(`[ProductPage] ⚠️ No products returned for gen=${gen}`);
        }
    };

    // ── Handle filter chip removal (✕ click) ──
    const handleRemoveOverride = async (key) => {
        const updated = { ...userOverrides };
        delete updated[key];
        setUserOverrides(updated);

        const gen = ++searchGenRef.current;
        console.log(`[ProductPage] 🗑️ Removed override '${key}', re-searching gen=${gen}`);
        setQueryLoading(true);
        try {
            await fireSearch(updated, gen);
        } catch (err) {
            console.error("Search after override removal failed:", err);
        } finally {
            setQueryLoading(false);
        }
    };

    const handleQuerySubmit = async (e) => {
        e.preventDefault();
        if (!userQuery.trim()) return;

        // ━━━ GUARD 1: Prevent double-firing ━━━
        if (queryInFlightRef.current) {
            console.log("[ProductPage] ⚠️ Query already in flight — skipping");
            return;
        }
        queryInFlightRef.current = true;

        const gen = ++searchGenRef.current;
        console.log(`[ProductPage] 🔄 Query submitted gen=${gen}: "${userQuery}"`);

        setQueryLoading(true);
        try {
            // Build context for LLM — use baseline + current overrides
            const currentAttributesForLLM = {
                category: item.class,
                color_name: visualBaseline?.color_name || attributes?.color_name,
                color_hex: visualBaseline?.color_hex || attributes?.color_hex,
                pattern: visualBaseline?.pattern || attributes?.pattern,
                sleeve_length: visualBaseline?.sleeve || attributes?.sleeve,
                original_color: visualBaseline?.color_name,
                original_pattern: visualBaseline?.pattern,
            };

            // 1. Call LLM — returns add/remove/reset_to_visual
            const llmRes = await axios.post('http://localhost:5000/llm', {
                user_query: userQuery,
                item: currentAttributesForLLM,
                scene: scene,
                session_history: sessionHistory || [],
            });

            const llmAdd = llmRes.data.add || llmRes.data.filters || {};
            const llmRemove = llmRes.data.remove || [];
            const llmReset = llmRes.data.reset_to_visual || false;
            const llmConfidence = llmRes.data.confidence || 0;
            const priceMax = llmRes.data.price_max || null;

            console.log(`[ProductPage] LLM response gen=${gen}:`, {
                add: llmAdd, remove: llmRemove, reset: llmReset,
                confidence: llmConfidence,
            });

            // ━━━ GUARD: Skip if LLM fallback/empty and no reset/remove ━━━
            if (!llmReset && llmRemove.length === 0 &&
                (!llmAdd || Object.keys(llmAdd).length === 0) &&
                llmConfidence < 0.1) {
                console.log(`[ProductPage] ⚠️ LLM FALLBACK — skipping search`);
                return;
            }

            // Staleness check after LLM wait
            if (searchGenRef.current !== gen) {
                console.log(`[ProductPage] ❌ Stale after LLM`);
                return;
            }

            // 2. Apply add/remove/reset to userOverrides
            let updatedOverrides;

            if (llmReset) {
                // RESET: clear all user overrides → PURE_SIMILARITY
                updatedOverrides = {};
                console.log(`[ProductPage] 🔄 RESET to visual baseline`);
            } else {
                updatedOverrides = { ...userOverrides };

                // Apply ADD (canonicalized, source-tagged)
                if (llmAdd && typeof llmAdd === 'object') {
                    for (const [rawKey, val] of Object.entries(llmAdd)) {
                        if (!val || val === '') continue;
                        const key = canonicalize(rawKey);
                        updatedOverrides[key] = { value: val, source: "user" };
                    }
                }

                // Apply REMOVE (canonicalized, with category guard)
                if (Array.isArray(llmRemove)) {
                    for (const rawKey of llmRemove) {
                        const key = canonicalize(rawKey);
                        // GUARD (Problem 2): category can only be removed if user overrode it
                        if (key === "category" && !userOverrides.category) {
                            console.log(`[ProductPage] ⛔ Blocked: cannot remove baseline category`);
                            continue;
                        }
                        delete updatedOverrides[key];
                    }
                }
            }

            // Store price_max as a user override (so it shows in chips and persists)
            if (priceMax !== null && priceMax !== undefined && !llmReset) {
                updatedOverrides.price = { value: priceMax, source: "user" };
            }

            setUserOverrides(updatedOverrides);

            // 3. Fire search with updated overrides
            await fireSearch(updatedOverrides, gen);

            setUserQuery("");
        } catch (err) {
            console.error("Query failed:", err);
            alert("Refinement failed. Try again.");
        } finally {
            queryInFlightRef.current = false;
            setQueryLoading(false);
        }
    };


    if (!item) return <div style={{ color: 'white', padding: 20 }}>No product selected.</div>;

    const mainProduct = products.length > 0 ? products[0] : null;
    const title = mainProduct
        ? mainProduct.name
        : `Searching for ${category.charAt(0).toUpperCase() + category.slice(1)}...`;
    const price = mainProduct ? mainProduct.price : null;

    return (
        <div style={styles.container}>
            {/* Header */}
            <header style={styles.header}>
                <div style={styles.backBar} onClick={() => navigate(-1)}>‹ Back</div>

                {/* AI Query Bar */}
                <form onSubmit={handleQuerySubmit} style={styles.queryForm}>
                    <input
                        type="text"
                        value={userQuery}
                        onChange={(e) => setUserQuery(e.target.value)}
                        placeholder="Refine... (e.g. 'blue one')"
                        style={styles.queryInput}
                    />
                    <button type="submit" disabled={queryLoading} style={styles.queryBtn}>
                        {queryLoading ? 'Thinking...' : 'Ask AI'}
                    </button>
                </form>

            </header>

            {/* ── Active Filter Chips (show ONLY user overrides) ── */}
            {Object.keys(userOverrides).length > 0 && (
                <div style={styles.filterChipsBar}>
                    <span style={{ fontSize: 12, color: '#666', marginRight: 8 }}>Active Filters:</span>
                    {Object.entries(userOverrides).map(([key, meta]) => (
                        <span key={key} style={styles.filterChip}>
                            {chipLabel(key)}: {key === 'price' ? `Under ₹${meta.value}` : meta.value}
                            <span
                                style={styles.filterChipX}
                                onClick={() => handleRemoveOverride(key)}
                                title={`Remove ${chipLabel(key)} filter`}
                            >✕</span>
                        </span>
                    ))}
                    <span
                        style={{ ...styles.filterChip, background: '#fff3cd', cursor: 'pointer' }}
                        onClick={() => {
                            setUserOverrides({});
                            const gen = ++searchGenRef.current;
                            setQueryLoading(true);
                            fireSearch({}, gen).finally(() => setQueryLoading(false));
                        }}
                    >🔄 Reset All</span>
                </div>
            )}

            {/* Layout: No Body Scroll, Internal Scroll */}
            <div style={styles.contentArea}>

                {/* Visual Analysis (Left) */}
                <div style={styles.visualColumn}>
                    {/* Detected Image (User Selection) */}
                    <div style={styles.imageContainer}>
                        <div style={styles.smallLabel}>Your Selection</div>
                        <img
                            src={`data:image/jpeg;base64,${item.cropped_image}`}
                            alt="Detected"
                            style={styles.detectedImage}
                        />
                    </div>
                </div>

                {/* Product Details (Right - Scrollable) */}
                <div style={styles.detailsColumn}>


                    {/* All Products — Uniform Grid */}
                    {products.length > 0 && (
                        <div style={styles.similarSection}>
                            <h3>Recommended Products</h3>
                            <div style={styles.similarGrid}>
                                {products.map((prod, idx) => (
                                    <div
                                        key={prod.id || idx}
                                        style={styles.similarCard}
                                        onClick={() => {
                                            setSelectedProduct(prod);
                                            trackEvent('product_click', {
                                                product_id: prod.product_id,
                                                rank: idx + 1,
                                                visual_similarity: prod.match_meta?.visual_similarity,
                                                final_score: prod.final_score,
                                            });
                                        }}
                                    >
                                        <div style={styles.similarImgWrapper}>
                                            <img
                                                src={prod.image_url}
                                                style={styles.similarImage}
                                                onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/100?text=No+Img" }}
                                            />
                                        </div>
                                        <div style={styles.similarName}>{prod.name}</div>
                                        <div style={{ fontWeight: '700', color: '#B12704', fontSize: 13 }}>₹{prod.price}</div>
                                        {prod.explanation && (
                                            <>
                                                <span
                                                    style={styles.whyBtn}
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        setExpandedExpl(prev => ({
                                                            ...prev,
                                                            [prod.product_id]: !prev[prod.product_id]
                                                        }));
                                                        if (!expandedExpl[prod.product_id]) {
                                                            trackEvent('explanation_view', {
                                                                product_id: prod.product_id,
                                                                explanation_shown: true,
                                                            });
                                                        }
                                                    }}
                                                >
                                                    💡 {expandedExpl[prod.product_id] ? 'Hide' : 'Why?'}
                                                </span>
                                                {expandedExpl[prod.product_id] && (
                                                    <div style={styles.explText}>{prod.explanation}</div>
                                                )}
                                            </>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* PRODUCT DETAIL MODAL */}
            {selectedProduct && (
                <div style={styles.modalOverlay} onClick={() => setSelectedProduct(null)}>
                    <div style={{ ...styles.modalContent, maxWidth: 600 }} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eee', paddingBottom: 10 }}>
                            <h2 style={{ margin: 0, fontSize: 18 }}>Product Details</h2>
                            <button onClick={() => setSelectedProduct(null)} style={{ border: 'none', background: 'none', fontSize: 20, cursor: 'pointer', color: '#555' }}>✕</button>
                        </div>
                        <div style={{ marginTop: 20, display: 'flex', gap: 30 }}>
                            <div style={{ flex: '0 0 240px', textAlign: 'center' }}>
                                <img
                                    src={selectedProduct.image_url}
                                    style={{ width: '100%', maxHeight: 320, objectFit: 'contain', borderRadius: 4 }}
                                    onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/240?text=No+Img" }}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <h3 style={{ margin: '0 0 10px', color: '#0F1111', fontSize: 20, lineHeight: 1.3 }}>{selectedProduct.name}</h3>
                                <div style={{ fontSize: 24, fontWeight: '500', color: '#B12704', marginBottom: 15 }}>₹{selectedProduct.price}</div>

                                <div style={{ fontSize: 14, lineHeight: 1.6, color: '#333' }}>
                                    {selectedProduct.brand && <div style={{ marginBottom: 5 }}><b>Brand:</b> {selectedProduct.brand}</div>}
                                    {selectedProduct.color && <div style={{ marginBottom: 5 }}><b>Color:</b> {selectedProduct.color}</div>}
                                    {selectedProduct.pattern && <div style={{ marginBottom: 5 }}><b>Pattern:</b> {selectedProduct.pattern}</div>}
                                    {selectedProduct.category && <div style={{ marginBottom: 5 }}><b>Category:</b> {selectedProduct.category}</div>}
                                </div>

                                {selectedProduct.final_score && (
                                    <div style={{ marginTop: 20, padding: 12, background: '#f0f2f2', borderRadius: 4, border: '1px solid #e7e7e7', fontSize: 13 }}>
                                        <b>Visual Match Score:</b> {Math.min(100, (selectedProduct.final_score * 100)).toFixed(0)}%
                                        <br />
                                        <span style={{ color: '#565959' }}>Based on deep learning embeddings</span>
                                    </div>
                                )}

                                <button
                                    onClick={() => {
                                        const fallbackParam = selectedProduct.name ? encodeURIComponent(selectedProduct.name) : selectedProduct.product_id;
                                        const url = selectedProduct.product_url || `https://www.myntra.com/${fallbackParam}`;
                                        trackEvent('buy_click', {
                                            product_id: selectedProduct.product_id,
                                            product_url: url,
                                            final_score: selectedProduct.final_score,
                                        });
                                        window.open(url, '_blank');
                                    }}
                                    style={{ marginTop: 20, padding: '10px 20px', background: '#FF3F6C', border: 'none', borderRadius: 20, cursor: 'pointer', fontWeight: 'bold', width: '100%', color: 'white', fontSize: 15 }}
                                >
                                    Buy Now
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
};

const styles = {
    container: { height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column', background: 'white', fontFamily: '"Amazon Ember", Arial, sans-serif' },
    header: { height: '60px', background: '#131921', display: 'flex', alignItems: 'center', padding: '0 20px', gap: 20, color: 'white' },
    backBar: { color: 'white', cursor: 'pointer', fontWeight: 'bold', fontSize: 14, display: 'flex', alignItems: 'center' },
    queryForm: { flex: 1, display: 'flex', maxWidth: 800, margin: '0 20px' },
    queryInput: { flex: 1, padding: '10px 15px', borderRadius: '4px 0 0 4px', border: 'none', outline: 'none', fontSize: 15 },
    queryBtn: { padding: '0 25px', borderRadius: '0 4px 4px 0', border: 'none', background: '#febd69', cursor: 'pointer', fontWeight: 'bold', color: '#111' },

    // Filter Chips Bar
    filterChipsBar: { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, padding: '8px 20px', background: '#f0f2f5', borderBottom: '1px solid #e7e7e7' },
    filterChip: { display: 'inline-flex', alignItems: 'center', gap: 4, padding: '4px 10px', background: '#e3f2fd', borderRadius: 16, fontSize: 12, fontWeight: 500, color: '#1565c0', border: '1px solid #bbdefb' },
    filterChipX: { cursor: 'pointer', fontSize: 14, color: '#c62828', marginLeft: 4, fontWeight: 'bold', lineHeight: 1 },

    contentArea: { flex: 1, display: 'flex', overflow: 'hidden', maxWidth: '1400px', margin: '0 auto', width: '100%' },

    // Left Column: Visuals
    visualColumn: {
        flex: '0 0 450px',
        background: 'white',
        padding: '30px 20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        borderRight: '1px solid #f0f0f0',
        overflowY: 'auto'
    },

    // Main Product (Best Match) - Top + Large
    matchContainer: {
        position: 'relative',
        width: '100%',
        border: 'none',
        padding: 10,
        cursor: 'pointer',
        textAlign: 'center',
        marginBottom: 30
    },
    matchImage: {
        maxWidth: '100%',
        maxHeight: '450px',
        width: 'auto',
        height: 'auto',
        objectFit: 'contain',
        transition: 'transform 0.2s'
    },
    matchLabelOverlay: {
        position: 'absolute',
        top: 0,
        left: 0,
        background: '#CC0C39', // Amazon "Best Seller" red looks professional
        color: 'white',
        padding: '4px 10px',
        fontSize: 12,
        fontWeight: 'bold',
        borderRadius: 2
    },

    // Detected Image (User Selection) - Bottom + Small
    imageContainer: {
        marginTop: 20,
        textAlign: 'center',
        borderTop: '1px solid #eee',
        paddingTop: 15,
        width: '100%'
    },
    smallLabel: { fontSize: 12, color: '#565959', marginBottom: 5, textTransform: 'uppercase', letterSpacing: 0.5 },
    detectedImage: {
        height: '100px',
        maxWidth: '100px',
        objectFit: 'cover',
        border: '1px solid #ddd',
        borderRadius: 4,
        boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
    },

    // Right Column: Details
    detailsColumn: { flex: 1, padding: '30px 40px', overflowY: 'auto' },
    title: { fontSize: 24, lineHeight: 1.3, margin: '0 0 10px 0', color: '#0F1111', fontWeight: 500 },
    ratingBlock: { marginBottom: 15, display: 'flex', alignItems: 'center' },
    priceRow: { marginBottom: 20, color: '#B12704', lineHeight: 1 },
    desc: { fontSize: 14, lineHeight: 1.6, color: '#333', marginBottom: 30 },

    similarSection: { borderTop: '1px solid #eee', paddingTop: 30, marginTop: 20 },
    similarGrid: { display: 'flex', gap: 20, flexWrap: 'wrap' },

    similarCard: {
        width: 160,
        border: '1px solid #eee',
        borderRadius: 8,
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'box-shadow 0.2s',
        padding: '10px'
    },
    similarImgWrapper: { height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 10 },
    similarImage: { maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' },
    similarName: { fontSize: 13, height: 38, overflow: 'hidden', marginBottom: 5, lineHeight: 1.4, color: '#007185' },

    modalOverlay: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 },
    modalContent: { background: 'white', padding: 30, borderRadius: 8, width: 600, maxWidth: '90%', maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 10px 25px rgba(0,0,0,0.2)' },
    scoreRow: { display: 'flex', gap: 20, marginBottom: 20 },
    scoreCard: { flex: 1, border: '1px solid #eee', padding: 15, borderRadius: 6, textAlign: 'center', background: '#FAFAFA' },

    // Explanation UI
    whyBtn: {
        display: 'inline-block', marginTop: 4, fontSize: 11, color: '#007185',
        cursor: 'pointer', fontWeight: 600, userSelect: 'none',
        padding: '2px 6px', borderRadius: 4, background: '#f0f9ff',
        transition: 'background 0.2s',
    },
    explText: {
        marginTop: 4, fontSize: 11, lineHeight: 1.5, color: '#555',
        fontStyle: 'italic', padding: '4px 6px', background: '#fafafa',
        borderRadius: 4, borderLeft: '2px solid #007185',
    },
};

export default ProductPage;
