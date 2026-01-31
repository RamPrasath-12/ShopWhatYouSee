
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MOVIES } from '../api/mockData';

// Icons
const SearchIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
    </svg>
);

const UserIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
        <circle cx="12" cy="7" r="4"></circle>
    </svg>
);

const PlayIconFilled = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 5V19L19 12L8 5Z" />
    </svg>
);

const InfoIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
    </svg>
);

const UploadIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="17 8 12 3 7 8"></polyline>
        <line x1="12" y1="3" x2="12" y2="15"></line>
    </svg>
);

import { fetchTrendingMovies, fetchTopRatedMovies, fetchActionMovies, fetchComedyMovies, fetchSciFiMovies } from '../api/tmdbClient';

const Browse = () => {
    const navigate = useNavigate();
    const [movies, setMovies] = useState([]);
    const [topRated, setTopRated] = useState([]);
    const [action, setAction] = useState([]);
    const [comedy, setComedy] = useState([]);
    const [scifi, setSciFi] = useState([]);
    const [heroMovie, setHeroMovie] = useState(null);

    useEffect(() => {
        const loadMovies = async () => {
            // Parallel fetch for speed
            const [trendingData, topRatedData, actionData, comedyData, scifiData] = await Promise.all([
                fetchTrendingMovies(),
                fetchTopRatedMovies(),
                fetchActionMovies(),
                fetchComedyMovies(),
                fetchSciFiMovies()
            ]);

            if (trendingData && trendingData.length > 0) {
                const shuffled = [...trendingData].sort(() => 0.5 - Math.random());
                setMovies(shuffled);
                // Keep original banner static as requested
            } else {
                setMovies(MOVIES);
            }

            if (topRatedData) setTopRated(topRatedData);
            if (actionData) setAction(actionData);
            if (comedyData) setComedy(comedyData);
            if (scifiData) setSciFi(scifiData);
        };
        loadMovies();
    }, []);

    const featuredMovie = heroMovie || MOVIES[0];

    const handlePlay = (movie) => {
        navigate(`/watch/${movie.id}`, { state: { movie } });
    };

    const handleVideoUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('video/')) {
            alert('Please select a video file');
            return;
        }

        // Create object URL for the video
        const videoUrl = URL.createObjectURL(file);

        // Navigate to Watch page with uploaded video
        navigate(`/watch/uploaded`, {
            state: {
                movie: {
                    id: 'uploaded',
                    title: file.name.replace(/\.[^/.]+$/, ''), // Remove extension
                    videoSrc: videoUrl,
                    isUploaded: true
                }
            }
        });
    };

    return (
        <div style={styles.container}>
            {/* Navbar - Prime Video Style */}
            <nav style={styles.navbar}>
                <div style={styles.leftNav}>
                    <div style={styles.logo}>
                        <span style={{ color: '#00a8e1' }}>prime</span> video
                    </div>
                    <div style={styles.navLinks}>
                        <span style={styles.activeLink}>Home</span>
                        <span style={styles.link}>Movies</span>
                        <span style={styles.link}>TV Shows</span>
                        <span style={styles.link}>Live TV</span>
                    </div>
                </div>
                <div style={styles.rightNav}>
                    {/* Hidden file input */}
                    <input
                        type="file"
                        accept="video/*"
                        onChange={handleVideoUpload}
                        style={{ display: 'none' }}
                        id="nav-upload-input"
                    />
                    {/* Upload button */}
                    <label htmlFor="nav-upload-input" style={styles.uploadButton}>
                        <UploadIcon />
                        <span style={{ marginLeft: '6px' }}>Upload Video</span>
                    </label>
                    <span style={styles.navIcon}><SearchIcon /></span>
                    <span style={styles.navText}>EN</span>
                    <span style={styles.navIcon}><UserIcon /></span>
                </div>
            </nav>

            {/* Hero Banner */}
            <div
                style={{
                    ...styles.hero,
                    backgroundImage: `linear-gradient(to right, rgba(0,0,0,0.8) 30%, transparent 70%), linear-gradient(to top, #1a1a1a 0%, transparent 50%), url(${featuredMovie.thumbnail})`
                }}
            >
                <div style={styles.heroContent}>
                    <div style={styles.primeBadge}>
                        <span style={{ color: '#00a8e1' }}>prime</span>
                    </div>
                    <h1 style={styles.heroTitle}>{featuredMovie.title}</h1>

                    {/* Language options */}
                    <div style={styles.languages}>
                        <span style={styles.langBadge}>English</span>
                        <span style={styles.separator}>|</span>
                        <span style={styles.langBadge}>Hindi</span>
                        <span style={styles.separator}>|</span>
                        <span style={styles.langBadge}>Tamil</span>
                    </div>

                    <div style={styles.ranking}>#1 in Fashion</div>

                    <div style={styles.heroButtons}>
                        <button style={styles.playButton} onClick={() => handlePlay(featuredMovie.id)}>
                            <PlayIconFilled />
                            <span>Play</span>
                        </button>
                        <button style={styles.addButton}>+ Add to watchlist</button>
                        <button style={styles.infoButton}>
                            <InfoIcon />
                            <span>Details</span>
                        </button>
                    </div>

                    <div style={styles.watchInfo}>
                        <span style={{ marginRight: '8px', fontSize: '18px' }}>✓</span>
                        Watch with a Prime membership
                    </div>
                </div>
            </div>



            {/* Movie Rows */}
            <div style={{ ...styles.contentSection, marginTop: '-80px', position: 'relative', zIndex: 10 }}>
                <div style={styles.rowHeader}>
                    <h3 style={styles.rowTitle}>Top Choices For You</h3>
                    <span style={styles.seeMore}>See more &gt;</span>
                </div>
                <div style={styles.row}>
                    {movies.map(movie => (
                        <div
                            key={movie.id}
                            style={styles.card}
                            onClick={() => handlePlay(movie)}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.transform = 'scale(1.05)';
                                e.currentTarget.style.zIndex = '10';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.transform = 'scale(1)';
                                e.currentTarget.style.zIndex = '1';
                            }}
                        >
                            <img src={movie.thumbnail} alt={movie.title} style={styles.cardImage} />
                            <div style={styles.primeBadgeCard}>prime</div>
                        </div>
                    ))}
                </div>

                {/* Additional Rows */}
                {topRated.length > 0 && (
                    <>
                        <div style={styles.rowHeader}>
                            <h3 style={styles.rowTitle}>Critically Acclaimed</h3>
                        </div>
                        <div style={styles.row}>
                            {topRated.map(movie => (
                                <div key={movie.id} style={styles.card} onClick={() => handlePlay(movie)}>
                                    <img src={movie.thumbnail} alt={movie.title} style={styles.cardImage} />
                                    <div style={styles.primeBadgeCard}>prime</div>
                                </div>
                            ))}
                        </div>
                    </>
                )}

                {action.length > 0 && (
                    <>
                        <div style={styles.rowHeader}>
                            <h3 style={styles.rowTitle}>Action Hits</h3>
                        </div>
                        <div style={styles.row}>
                            {action.map(movie => (
                                <div key={movie.id} style={styles.card} onClick={() => handlePlay(movie)}>
                                    <img src={movie.thumbnail} alt={movie.title} style={styles.cardImage} />
                                    <div style={styles.primeBadgeCard}>prime</div>
                                </div>
                            ))}
                        </div>
                    </>
                )}

                {comedy.length > 0 && (
                    <>
                        <div style={styles.rowHeader}>
                            <h3 style={styles.rowTitle}>Comedy Favorites</h3>
                        </div>
                        <div style={styles.row}>
                            {comedy.map(movie => (
                                <div key={movie.id} style={styles.card} onClick={() => handlePlay(movie)}>
                                    <img src={movie.thumbnail} alt={movie.title} style={styles.cardImage} />
                                    <div style={styles.primeBadgeCard}>prime</div>
                                </div>
                            ))}
                        </div>
                    </>
                )}

                {scifi.length > 0 && (
                    <>
                        <div style={styles.rowHeader}>
                            <h3 style={styles.rowTitle}>Sci-Fi & Fantasy</h3>
                        </div>
                        <div style={styles.row}>
                            {scifi.map(movie => (
                                <div key={movie.id} style={styles.card} onClick={() => handlePlay(movie)}>
                                    <img src={movie.thumbnail} alt={movie.title} style={styles.cardImage} />
                                    <div style={styles.primeBadgeCard}>prime</div>
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div >
    );
};

const styles = {
    container: {
        backgroundColor: "#1a1a1a",
        minHeight: "100vh",
        color: "white",
        fontFamily: "Arial, sans-serif"
    },
    navbar: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 30px",
        background: "#0f171e",
        position: "fixed",
        width: "100%",
        top: 0,
        zIndex: 100,
        boxSizing: "border-box"
    },
    leftNav: {
        display: "flex",
        alignItems: "center",
        gap: "40px"
    },
    logo: {
        fontSize: "20px",
        fontWeight: "600",
        color: "white"
    },
    navLinks: {
        display: "flex",
        gap: "25px",
        fontSize: "15px",
        fontWeight: "600"
    },
    activeLink: {
        color: "white",
        cursor: "pointer",
        fontWeight: "700",
        borderBottom: "2px solid white",
        paddingBottom: "5px"
    },
    link: {
        color: "#aaa",
        cursor: "pointer",
        transition: "color 0.2s",
        fontWeight: "600"
    },
    rightNav: {
        display: "flex",
        gap: "25px",
        alignItems: "center",
        fontSize: "16px"
    },
    navIcon: {
        cursor: "pointer",
        color: "#ccc",
        display: "flex",
        alignItems: "center"
    },
    navText: {
        cursor: "pointer",
        color: "#ccc",
        fontWeight: "600"
    },
    hero: {
        height: "85vh",
        backgroundSize: "cover",
        backgroundPosition: "center top",
        position: "relative",
        display: "flex",
        alignItems: "center",
        paddingLeft: "60px",
        marginTop: "0"
    },
    heroContent: {
        maxWidth: "600px",
        marginTop: "80px"
    },
    primeBadge: {
        fontSize: "18px",
        fontWeight: "bold",
        marginBottom: "10px",
        textTransform: "uppercase",
        letterSpacing: "1px"
    },
    heroTitle: {
        fontSize: "64px",
        marginBottom: "20px",
        fontWeight: "800",
        lineHeight: "1.1",
        textShadow: "2px 2px 10px rgba(0,0,0,0.5)"
    },
    languages: {
        fontSize: "15px",
        marginBottom: "15px",
        color: "#ccc",
        fontWeight: "500"
    },
    langBadge: {
        color: "#ddd"
    },
    separator: {
        margin: "0 10px",
        color: "#666"
    },
    ranking: {
        fontSize: "16px",
        color: "#fff",
        marginBottom: "30px",
        fontWeight: "600"
    },
    heroButtons: {
        display: "flex",
        gap: "15px",
        marginBottom: "20px",
        alignItems: "center"
    },
    playButton: {
        padding: "16px 32px",
        fontSize: "18px",
        backgroundColor: "#fff",
        color: "#0f171e",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        fontWeight: "700",
        display: "flex",
        alignItems: "center",
        gap: "10px",
        transition: "transform 0.2s"
    },
    addButton: {
        padding: "16px 28px",
        fontSize: "18px",
        backgroundColor: "rgba(255,255,255,0.2)",
        color: "white",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        fontWeight: "600",
        backdropFilter: "blur(5px)"
    },
    infoButton: {
        padding: "16px 28px",
        fontSize: "18px",
        backgroundColor: "rgba(255,255,255,0.2)",
        color: "white",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        fontWeight: "600",
        backdropFilter: "blur(5px)",
        display: "flex",
        alignItems: "center",
        gap: "10px"
    },
    watchInfo: {
        fontSize: "15px",
        color: "#fff",
        display: "flex",
        alignItems: "center",
        fontWeight: "500"
    },
    contentSection: {
        padding: "40px 60px 80px 60px",
        marginTop: "-100px",
        position: "relative",
        zIndex: 10
    },
    rowHeader: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "20px",
        marginTop: "50px"
    },
    rowTitle: {
        fontSize: "22px",
        fontWeight: "700",
        margin: 0,
        color: "#fff"
    },
    seeMore: {
        fontSize: "14px",
        color: "#79b8f3",
        cursor: "pointer",
        fontWeight: "600"
    },
    row: {
        display: "flex",
        gap: "20px",
        overflowX: "auto",
        paddingBottom: "30px",
        scrollbarWidth: "none"
    },
    card: {
        minWidth: "280px",
        height: "158px",
        borderRadius: "6px",
        overflow: "hidden",
        cursor: "pointer",
        position: "relative",
        transition: "transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        boxShadow: "0 4px 15px rgba(0,0,0,0.5)",
        zIndex: 1
    },
    cardImage: {
        width: "100%",
        height: "100%",
        objectFit: "cover"
    },
    primeBadgeCard: {
        position: "absolute",
        top: "0",
        left: "0",
        background: "#00a8e1",
        color: "white",
        padding: "2px 6px",
        borderBottomRightRadius: "4px",
        fontSize: "11px",
        fontWeight: "800",
        textTransform: "uppercase"
    },

    uploadButton: {
        display: 'flex',
        alignItems: 'center',
        padding: '8px 16px',
        background: 'rgba(0, 168, 225, 0.15)',
        border: '1px solid #00a8e1',
        borderRadius: '6px',
        color: 'white',
        fontSize: '14px',
        fontWeight: '500',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        marginRight: '15px',
        '&:hover': {
            background: 'rgba(0, 168, 225, 0.25)',
            borderColor: '#00c9ff'
        }
    }
};

export default Browse;
