import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Insights = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        axios.get('http://localhost:5000/insights')
            .then(res => {
                setData(res.data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to load insights", err);
                setLoading(false);
            });
    }, []);

    if (loading) return <div style={styles.container}>Loading Insights...</div>;
    if (!data) return <div style={styles.container}>No Data Available</div>;

    const { stats, analysis, session_data } = data;
    const { relevance_level, successful_recommendation, strengths, weaknesses, improvement_suggestion, user_behavior_analysis } = analysis || {};

    return (
        <div style={styles.container}>
            <header style={styles.header}>
                <h1 style={{ margin: 0 }}>System Evaluation & Insights</h1>
                <div style={{ fontSize: '14px', opacity: 0.8 }}>Latest Session Analysis</div>
            </header>

            <div style={styles.grid}>
                {/* GLOBAL STATS */}
                <div style={styles.card}>
                    <h2>Global Performance</h2>
                    <div style={styles.statRow}>
                        <span>Avg Satisfaction:</span>
                        <span style={styles.statValue}>{stats.average_rating} ⭐</span>
                    </div>
                    <div style={styles.statRow}>
                        <span>Total Sessions:</span>
                        <span style={styles.statValue}>{stats.total_ratings}</span>
                    </div>

                    <h3>Platform Performance</h3>
                    {stats.insights.llm_performance && Object.entries(stats.insights.llm_performance).map(([source, data]) => (
                        <div key={source} style={styles.statRowSmall}>
                            <span>{source}:</span>
                            <span>{data.avg_rating ? data.avg_rating.toFixed(2) : 'N/A'} ({data.count})</span>
                        </div>
                    ))}
                </div>

                {/* LATEST SESSION EVALUATION */}
                <div style={{ ...styles.card, flex: 2, borderLeft: '5px solid #00a8e1' }}>
                    <h2>Latest Session Evaluation (LLM Agent)</h2>

                    <div style={styles.evaluationHeader}>
                        <div style={styles.badge(relevance_level)}>Relevance: {relevance_level || 'N/A'}</div>
                        <div style={styles.badge(successful_recommendation ? 'High' : 'Low')}>
                            Success: {successful_recommendation ? 'YES' : 'NO'}
                        </div>
                    </div>

                    <div style={styles.section}>
                        <h3>Strengths</h3>
                        <ul>
                            {strengths && strengths.map((s, i) => <li key={i}>{s}</li>)}
                        </ul>
                    </div>

                    <div style={styles.section}>
                        <h3>Weaknesses & Risks</h3>
                        <ul>
                            {weaknesses && weaknesses.map((w, i) => <li key={i} style={{ color: '#d9534f' }}>{w}</li>)}
                        </ul>
                    </div>

                    <div style={styles.section}>
                        <h3>Technical Improvement</h3>
                        <div style={styles.improvementBox}>
                            {improvement_suggestion || "No suggestion provided."}
                        </div>
                    </div>

                    <div style={styles.section}>
                        <h3>User Behavior Analysis</h3>
                        <p>{user_behavior_analysis}</p>
                    </div>
                </div>
            </div>

            <div style={styles.raw}>
                <h3>Raw Session Data</h3>
                <pre>{JSON.stringify(session_data, null, 2)}</pre>
            </div>
        </div>
    );
};

const styles = {
    container: {
        padding: '20px',
        fontFamily: "'Segoe UI', Arial, sans-serif",
        backgroundColor: '#f4f6f8',
        minHeight: '100vh',
    },
    header: {
        marginBottom: '20px',
        paddingBottom: '10px',
        borderBottom: '1px solid #ddd'
    },
    grid: {
        display: 'flex',
        gap: '20px',
        flexWrap: 'wrap'
    },
    card: {
        backgroundColor: 'white',
        padding: '20px',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        flex: 1,
        minWidth: '300px'
    },
    statRow: {
        display: 'flex',
        justifyContent: 'space-between',
        marginBottom: '10px',
        fontSize: '18px',
        fontWeight: 'bold'
    },
    statValue: {
        color: '#007185'
    },
    statRowSmall: {
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '14px',
        marginBottom: '5px'
    },
    evaluationHeader: {
        display: 'flex',
        gap: '10px',
        marginBottom: '20px'
    },
    badge: (type) => ({
        padding: '5px 12px',
        borderRadius: '15px',
        backgroundColor: type === 'High' ? '#dff0d8' : type === 'Low' ? '#f2dede' : '#fcf8e3',
        color: type === 'High' ? '#3c763d' : type === 'Low' ? '#a94442' : '#8a6d3b',
        fontWeight: 'bold',
        textTransform: 'uppercase',
        fontSize: '12px'
    }),
    section: {
        marginBottom: '15px'
    },
    improvementBox: {
        padding: '10px',
        backgroundColor: '#e8f4fd',
        borderLeft: '4px solid #00a8e1',
        fontStyle: 'italic'
    },
    raw: {
        marginTop: '30px',
        padding: '10px',
        backgroundColor: '#eee',
        borderRadius: '4px',
        fontSize: '12px',
        overflowX: 'auto'
    }
};

export default Insights;
