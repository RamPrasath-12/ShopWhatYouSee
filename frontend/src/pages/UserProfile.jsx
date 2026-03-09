
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';



const COLORS = [
    'Black', 'White', 'Blue', 'Red', 'Green', 'Pink', 'Yellow',
    'Grey', 'Navy Blue', 'Maroon', 'Brown', 'Purple', 'Orange', 'Beige',
];

const STYLES = ['Casual', 'Formal', 'Sports', 'Party', 'Ethnic'];

const SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];

// ─────────────────────────────────────────────────
// MULTI-USER LOGIN & PROFILE PAGE
// Supports multiple users so you can demo personalized
// recommendations for different people on the same product.
// ─────────────────────────────────────────────────

const UserProfile = () => {
    const navigate = useNavigate();

    // ── Multi-user state ──
    const [allUsers, setAllUsers] = useState(() => {
        try {
            return JSON.parse(localStorage.getItem('swys_all_users') || '[]');
        } catch { return []; }
    });

    const [userId, setUserId] = useState(() => {
        return localStorage.getItem('swys_user_id') || '';
    });
    const [name, setName] = useState(() => {
        return localStorage.getItem('swys_user_name') || '';
    });

    // ── Preference form state ──
    const [gender, setGender] = useState('');
    const [size, setSize] = useState('M');
    const [budgetMin, setBudgetMin] = useState(0);
    const [budgetMax, setBudgetMax] = useState(5000);

    const [selectedColors, setSelectedColors] = useState([]);
    const [selectedStyles, setSelectedStyles] = useState([]);
    const [selectedBrands, setSelectedBrands] = useState([]);
    const [brandsByCategory, setBrandsByCategory] = useState({});
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [loadingProfile, setLoadingProfile] = useState(false);

    // ── New user form (for creating additional users) ──
    const [showNewUser, setShowNewUser] = useState(false);
    const [newUserName, setNewUserName] = useState('');

    // Load brands by category on mount
    useEffect(() => {
        axios.get('http://localhost:5000/brands-by-category')
            .then(res => {
                setBrandsByCategory(res.data.brands_by_category || {});
            })
            .catch(() => { });
    }, []);

    // Load existing preferences when userId changes
    useEffect(() => {
        if (userId) {
            setLoadingProfile(true);
            axios.get(`http://localhost:5000/user/preferences?user_id=${userId}`)
                .then(res => {
                    const prefs = res.data.preferences;
                    if (prefs && Object.keys(prefs).length > 0) {
                        setGender(prefs.gender || '');
                        setSize(prefs.size || 'M');
                        setBudgetMin(prefs.budget_min || 0);
                        setBudgetMax(prefs.budget_max || 5000);

                        setSelectedColors(prefs.preferred_colors || []);
                        setSelectedStyles(prefs.preferred_styles || []);
                        setSelectedBrands(prefs.preferred_brands || []);
                    } else {
                        // New user — reset form
                        setGender('');
                        setSize('M');
                        setBudgetMin(0);
                        setBudgetMax(5000);

                        setSelectedColors([]);
                        setSelectedStyles([]);
                        setSelectedBrands([]);
                    }
                })
                .catch(() => { })
                .finally(() => setLoadingProfile(false));
        }
    }, [userId]);

    const toggleInArray = (arr, item) => {
        return arr.includes(item) ? arr.filter(x => x !== item) : [...arr, item];
    };

    // ── Switch to existing user ──
    const switchUser = (user) => {
        setUserId(user.id);
        setName(user.name);
        localStorage.setItem('swys_user_id', user.id);
        localStorage.setItem('swys_user_name', user.name);
        setSaved(false);
    };

    // ── Create new user ──
    const createNewUser = () => {
        if (!newUserName.trim()) return;
        const newId = `user_${Date.now()}`;
        const newUser = { id: newId, name: newUserName.trim() };

        const updatedUsers = [...allUsers, newUser];
        setAllUsers(updatedUsers);
        localStorage.setItem('swys_all_users', JSON.stringify(updatedUsers));

        // Switch to new user
        switchUser(newUser);
        setShowNewUser(false);
        setNewUserName('');
    };

    // ── Logout ──
    const handleLogout = () => {
        localStorage.removeItem('swys_user_id');
        localStorage.removeItem('swys_user_name');
        setUserId('');
        setName('');
        setGender('');

        setSelectedColors([]);
        setSelectedStyles([]);
        setSelectedBrands([]);
    };

    // ── Save preferences ──
    const handleSave = async () => {
        if (!name.trim()) {
            alert('Please enter your name');
            return;
        }

        setSaving(true);
        const finalUserId = userId || `user_${Date.now()}`;

        try {
            await axios.post('http://localhost:5000/user/preferences', {
                user_id: finalUserId,
                name: name.trim(),
                gender,
                size,
                budget_min: budgetMin,
                budget_max: budgetMax,

                preferred_colors: selectedColors,
                preferred_styles: selectedStyles,
                preferred_brands: selectedBrands,
            });

            localStorage.setItem('swys_user_id', finalUserId);
            localStorage.setItem('swys_user_name', name.trim());
            setUserId(finalUserId);

            // Add to allUsers if not exists
            const exists = allUsers.find(u => u.id === finalUserId);
            if (!exists) {
                const updatedUsers = [...allUsers, { id: finalUserId, name: name.trim() }];
                setAllUsers(updatedUsers);
                localStorage.setItem('swys_all_users', JSON.stringify(updatedUsers));
            } else {
                // Update name if changed
                const updatedUsers = allUsers.map(u =>
                    u.id === finalUserId ? { ...u, name: name.trim() } : u
                );
                setAllUsers(updatedUsers);
                localStorage.setItem('swys_all_users', JSON.stringify(updatedUsers));
            }

            setSaved(true);
            setTimeout(() => setSaved(false), 3000);
        } catch (err) {
            console.error('Failed to save preferences:', err);
            alert('Failed to save. Check if backend is running.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div style={styles.page}>
            {/* Navbar */}
            <nav style={styles.navbar}>
                <div style={styles.logo} onClick={() => navigate('/')}>
                    <span style={{ color: '#00a8e1' }}>prime</span> video
                </div>
                <div style={{ fontSize: 16, fontWeight: 600 }}>👤 User Profiles</div>
                <div style={styles.navRight}>
                    <span style={styles.backLink} onClick={() => navigate('/')}>← Back to Home</span>
                </div>
            </nav>

            <div style={styles.content}>

                {/* ═══════════════════════════════════════════════════
                    MULTI-USER SWITCHER (visible always)
                ═══════════════════════════════════════════════════ */}
                <div style={styles.userSwitcher}>
                    <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#aab8c2' }}>
                        Switch User
                    </h3>
                    <div style={styles.userCards}>
                        {allUsers.map(user => (
                            <div
                                key={user.id}
                                style={{
                                    ...styles.userCard,
                                    ...(userId === user.id ? styles.userCardActive : {}),
                                }}
                                onClick={() => switchUser(user)}
                            >
                                <div style={{
                                    ...styles.userAvatar,
                                    background: userId === user.id
                                        ? 'linear-gradient(135deg, #00a8e1, #0066cc)'
                                        : '#2a3a4a',
                                }}>
                                    {user.name.charAt(0).toUpperCase()}
                                </div>
                                <span style={{ fontSize: 13, fontWeight: userId === user.id ? 700 : 400 }}>
                                    {user.name}
                                </span>
                                {userId === user.id && (
                                    <span style={{ fontSize: 10, color: '#00a8e1' }}>● Active</span>
                                )}
                            </div>
                        ))}

                        {/* Add New User Button */}
                        <div
                            style={{ ...styles.userCard, borderStyle: 'dashed' }}
                            onClick={() => setShowNewUser(true)}
                        >
                            <div style={{ ...styles.userAvatar, background: '#1a2634', fontSize: 20 }}>+</div>
                            <span style={{ fontSize: 12, color: '#667' }}>New User</span>
                        </div>
                    </div>

                    {/* New User Form */}
                    {showNewUser && (
                        <div style={{ display: 'flex', gap: 10, marginTop: 12, alignItems: 'center' }}>
                            <input
                                type="text"
                                value={newUserName}
                                onChange={e => setNewUserName(e.target.value)}
                                placeholder="Enter new user's name"
                                style={{ ...styles.input, flex: 1, maxWidth: 300 }}
                                onKeyDown={e => e.key === 'Enter' && createNewUser()}
                                autoFocus
                            />
                            <button onClick={createNewUser} style={styles.smallBtn}>Create</button>
                            <button
                                onClick={() => { setShowNewUser(false); setNewUserName(''); }}
                                style={{ ...styles.smallBtn, background: '#444' }}
                            >Cancel</button>
                        </div>
                    )}

                    {userId && (
                        <button onClick={handleLogout} style={{ ...styles.smallBtn, background: '#8b0000', marginTop: 10 }}>
                            Logout
                        </button>
                    )}
                </div>

                {/* ═══════════════════════════════════════════════════
                    PREFERENCE FORM (shown when a user is selected)
                ═══════════════════════════════════════════════════ */}
                {userId ? (
                    <>
                        {/* Profile Header */}
                        <div style={styles.profileHeader}>
                            <div style={styles.avatar}>
                                {name ? name.charAt(0).toUpperCase() : '?'}
                            </div>
                            <div>
                                <h1 style={styles.heading}>{name}'s Fashion Preferences</h1>
                                <p style={styles.subtitle}>
                                    These preferences personalize your product recommendations
                                </p>
                            </div>
                        </div>

                        {saved && (
                            <div style={styles.successBanner}>
                                ✅ Preferences saved for <b>{name}</b>! Recommendations are now personalized.
                            </div>
                        )}

                        {loadingProfile ? (
                            <div style={{ textAlign: 'center', padding: 40, color: '#8899a6' }}>Loading preferences...</div>
                        ) : (
                            <>
                                {/* Form */}
                                <div style={styles.formGrid}>
                                    {/* Name */}
                                    <div style={styles.fieldGroup}>
                                        <label style={styles.label}>Name</label>
                                        <input
                                            type="text"
                                            value={name}
                                            onChange={e => setName(e.target.value)}
                                            placeholder="Your name"
                                            style={styles.input}
                                        />
                                    </div>

                                    {/* Gender */}
                                    <div style={styles.fieldGroup}>
                                        <label style={styles.label}>Gender</label>
                                        <div style={styles.chipRow}>
                                            {['Men', 'Women', 'Unisex'].map(g => (
                                                <span
                                                    key={g}
                                                    style={{
                                                        ...styles.chip,
                                                        ...(gender === g ? styles.chipSelected : {}),
                                                    }}
                                                    onClick={() => setGender(g)}
                                                >
                                                    {g}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Size */}
                                    <div style={styles.fieldGroup}>
                                        <label style={styles.label}>Clothing Size</label>
                                        <div style={styles.chipRow}>
                                            {SIZES.map(s => (
                                                <span
                                                    key={s}
                                                    style={{
                                                        ...styles.chip,
                                                        ...(size === s ? styles.chipSelected : {}),
                                                    }}
                                                    onClick={() => setSize(s)}
                                                >
                                                    {s}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Budget */}
                                    <div style={styles.fieldGroup}>
                                        <label style={styles.label}>Budget Range</label>
                                        <div style={{ display: 'flex', gap: 15, alignItems: 'center' }}>
                                            <div>
                                                <span style={styles.smallLabel}>Min ₹</span>
                                                <input
                                                    type="number"
                                                    value={budgetMin}
                                                    onChange={e => setBudgetMin(Number(e.target.value))}
                                                    style={{ ...styles.input, width: 120 }}
                                                />
                                            </div>
                                            <span style={{ color: '#666', marginTop: 20 }}>—</span>
                                            <div>
                                                <span style={styles.smallLabel}>Max ₹</span>
                                                <input
                                                    type="number"
                                                    value={budgetMax}
                                                    onChange={e => setBudgetMax(Number(e.target.value))}
                                                    style={{ ...styles.input, width: 120 }}
                                                />
                                            </div>
                                        </div>
                                    </div>



                                    {/* Colors */}
                                    <div style={{ ...styles.fieldGroup, gridColumn: '1 / -1' }}>
                                        <label style={styles.label}>Preferred Colors <span style={styles.optional}>(optional)</span></label>
                                        <div style={styles.chipRow}>
                                            {COLORS.map(color => (
                                                <span
                                                    key={color}
                                                    style={{
                                                        ...styles.chip,
                                                        ...(selectedColors.includes(color) ? styles.chipSelected : {}),
                                                    }}
                                                    onClick={() => setSelectedColors(toggleInArray(selectedColors, color))}
                                                >
                                                    {color}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Styles */}
                                    <div style={{ ...styles.fieldGroup, gridColumn: '1 / -1' }}>
                                        <label style={styles.label}>Preferred Styles <span style={styles.optional}>(optional)</span></label>
                                        <div style={styles.chipRow}>
                                            {STYLES.map(s => (
                                                <span
                                                    key={s}
                                                    style={{
                                                        ...styles.chip,
                                                        ...(selectedStyles.includes(s) ? styles.chipSelected : {}),
                                                    }}
                                                    onClick={() => setSelectedStyles(toggleInArray(selectedStyles, s))}
                                                >
                                                    {s}
                                                </span>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Preferred Brands */}
                                    <div style={{ ...styles.fieldGroup, gridColumn: '1 / -1' }}>
                                        <label style={styles.label}>Preferred Brands <span style={styles.optional}>(optional)</span></label>
                                        <div style={{ fontSize: 12, color: '#8899a6', marginBottom: 8 }}>Select brands by category</div>
                                        {Object.keys(brandsByCategory).length > 0 ? (
                                            <div style={{ maxHeight: 200, overflowY: 'auto', border: '1px solid #333', borderRadius: 8, padding: 10, background: '#0d1117' }}>
                                                {Object.entries(brandsByCategory).map(([cat, brands]) => (
                                                    <div key={cat} style={{ marginBottom: 10 }}>
                                                        <div style={{ fontSize: 12, fontWeight: 700, color: '#58a6ff', marginBottom: 4, textTransform: 'capitalize' }}>
                                                            {cat}
                                                        </div>
                                                        <div style={styles.chipRow}>
                                                            {brands.map(brand => (
                                                                <span
                                                                    key={`${cat}-${brand}`}
                                                                    style={{
                                                                        ...styles.chip,
                                                                        fontSize: 11,
                                                                        padding: '4px 10px',
                                                                        ...(selectedBrands.includes(brand) ? styles.chipSelected : {}),
                                                                    }}
                                                                    onClick={() => setSelectedBrands(toggleInArray(selectedBrands, brand))}
                                                                >
                                                                    {brand}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div style={{ fontSize: 12, color: '#666' }}>Loading brands...</div>
                                        )}
                                        {selectedBrands.length > 0 && (
                                            <div style={{ marginTop: 6, fontSize: 12, color: '#aaa' }}>
                                                Selected: {selectedBrands.join(', ')}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Save Button */}
                                <button
                                    style={{
                                        ...styles.saveBtn,
                                        opacity: saving ? 0.6 : 1,
                                    }}
                                    onClick={handleSave}
                                    disabled={saving}
                                >
                                    {saving ? 'Saving...' : '💾 Save Preferences'}
                                </button>
                            </>
                        )}
                    </>
                ) : (
                    /* No user selected prompt */
                    <div style={{ textAlign: 'center', padding: '60px 0', color: '#8899a6' }}>
                        <div style={{ fontSize: 48, marginBottom: 20 }}>👤</div>
                        <h2 style={{ color: 'white', margin: '0 0 10px' }}>No User Selected</h2>
                        <p>Create a new user or select an existing one above to set preferences.</p>
                    </div>
                )}
            </div>
        </div>
    );
};


const styles = {
    page: {
        minHeight: '100vh',
        background: '#0f171e',
        color: '#e0e0e0',
        fontFamily: "'Amazon Ember', Arial, sans-serif",
    },
    navbar: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 30px',
        background: '#0f171e',
        borderBottom: '1px solid #1a2634',
        position: 'sticky',
        top: 0,
        zIndex: 100,
    },
    logo: {
        fontSize: 20,
        fontWeight: 600,
        color: 'white',
        cursor: 'pointer',
    },
    navRight: {
        display: 'flex',
        gap: 20,
        alignItems: 'center',
    },
    backLink: {
        color: '#00a8e1',
        cursor: 'pointer',
        fontSize: 14,
        fontWeight: 500,
    },
    content: {
        maxWidth: 800,
        margin: '0 auto',
        padding: '30px 30px 80px',
    },

    // ── Multi-User Switcher ──
    userSwitcher: {
        marginBottom: 30,
        padding: 20,
        background: '#141e28',
        borderRadius: 12,
        border: '1px solid #1e2e3e',
    },
    userCards: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: 12,
    },
    userCard: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 6,
        padding: '12px 16px',
        borderRadius: 10,
        border: '1px solid #2a3a4a',
        background: '#1a2634',
        cursor: 'pointer',
        transition: 'all 0.2s',
        minWidth: 80,
    },
    userCardActive: {
        borderColor: '#00a8e1',
        background: '#0d2236',
        boxShadow: '0 0 10px rgba(0, 168, 225, 0.2)',
    },
    userAvatar: {
        width: 42,
        height: 42,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 18,
        fontWeight: 700,
        color: 'white',
    },
    smallBtn: {
        padding: '6px 16px',
        fontSize: 13,
        fontWeight: 600,
        background: '#00a8e1',
        color: 'white',
        border: 'none',
        borderRadius: 6,
        cursor: 'pointer',
    },

    // ── Profile Form ──
    profileHeader: {
        display: 'flex',
        alignItems: 'center',
        gap: 20,
        marginBottom: 30,
    },
    avatar: {
        width: 70,
        height: 70,
        borderRadius: '50%',
        background: 'linear-gradient(135deg, #00a8e1, #0066cc)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 28,
        fontWeight: 700,
        color: 'white',
        textTransform: 'uppercase',
    },
    heading: {
        fontSize: 24,
        fontWeight: 700,
        margin: '0 0 4px',
        color: 'white',
    },
    subtitle: {
        fontSize: 14,
        color: '#8899a6',
        margin: 0,
    },
    successBanner: {
        background: '#1a3a2a',
        border: '1px solid #2d6b4f',
        color: '#4caf50',
        padding: '12px 20px',
        borderRadius: 8,
        marginBottom: 25,
        fontSize: 14,
        fontWeight: 500,
    },
    formGrid: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 25,
    },
    fieldGroup: {
        marginBottom: 5,
    },
    label: {
        display: 'block',
        fontSize: 14,
        fontWeight: 600,
        marginBottom: 8,
        color: '#c0c8d0',
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    smallLabel: {
        display: 'block',
        fontSize: 11,
        color: '#8899a6',
        marginBottom: 4,
    },
    optional: {
        fontSize: 11,
        fontWeight: 400,
        color: '#667',
        textTransform: 'none',
    },
    input: {
        width: '100%',
        padding: '10px 14px',
        borderRadius: 6,
        border: '1px solid #2a3a4a',
        background: '#1a2634',
        color: '#e0e0e0',
        fontSize: 15,
        outline: 'none',
        boxSizing: 'border-box',
    },
    chipRow: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: 8,
    },
    chip: {
        padding: '6px 14px',
        borderRadius: 20,
        border: '1px solid #2a3a4a',
        background: '#1a2634',
        color: '#aab8c2',
        fontSize: 13,
        cursor: 'pointer',
        transition: 'all 0.2s',
        userSelect: 'none',
    },
    chipSelected: {
        background: '#00a8e1',
        borderColor: '#00a8e1',
        color: 'white',
        fontWeight: 600,
    },
    saveBtn: {
        marginTop: 30,
        padding: '14px 40px',
        fontSize: 16,
        fontWeight: 700,
        background: 'linear-gradient(135deg, #00a8e1, #0077b5)',
        color: 'white',
        border: 'none',
        borderRadius: 8,
        cursor: 'pointer',
        width: '100%',
        transition: 'transform 0.2s, box-shadow 0.2s',
        boxShadow: '0 4px 15px rgba(0, 168, 225, 0.3)',
    },
};

export default UserProfile;
