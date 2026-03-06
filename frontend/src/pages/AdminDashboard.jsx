import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
    BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const API = 'http://localhost:5000/api/admin';

// ─── Color palette ───
const COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#ec4899', '#14b8a6'];
const C = { primary: '#6366f1', secondary: '#22d3ee', accent: '#f59e0b', danger: '#ef4444', success: '#10b981' };

// ─── API helper ───
const api = (path, token, opts = {}) => {
    const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
    if (opts.method === 'POST') return axios.post(`${API}${path}`, opts.body || {}, { headers });
    return axios.get(`${API}${path}`, { headers, params: opts.params });
};

// ─── Tab config ───
const TABS = [
    { id: 'kpis', label: 'KPI Overview', icon: '📈' },
    { id: 'health', label: 'System Health', icon: '🤖' },
    { id: 'quality', label: 'Rec. Quality', icon: '🎯' },
    { id: 'business', label: 'Business', icon: '💼' },
    { id: 'behaviour', label: 'User Behaviour', icon: '👤' },
    { id: 'trends', label: 'Trends', icon: '📉' },
    { id: 'events', label: 'Raw Events', icon: '🔎' },
    { id: 'insights', label: 'LLM Insights', icon: '🧠' },
];

// ═══════════════════════════════════════════════════
// LOGIN SCREEN
// ═══════════════════════════════════════════════════
const LoginScreen = ({ onLogin }) => {
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(''); setLoading(true);
        try {
            const res = await axios.post(`${API}/login`, { password });
            onLogin(res.data.token);
        } catch { setError('Invalid password'); }
        setLoading(false);
    };

    return (
        <div style={s.loginContainer}>
            <form onSubmit={handleSubmit} style={s.loginBox}>
                <div style={{ fontSize: 40, marginBottom: 16 }}>🔒</div>
                <h2 style={{ fontSize: 20, fontWeight: 600, color: '#f1f5f9', marginBottom: 24 }}>ShopWhatYouSee Admin</h2>
                <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter admin password" style={s.loginInput} autoFocus />
                {error && <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 12 }}>{error}</div>}
                <button type="submit" style={s.primaryBtn} disabled={loading}>{loading ? 'Authenticating...' : 'Login'}</button>
            </form>
        </div>
    );
};

// ═══════════════════════════════════════════════════
// REUSABLE COMPONENTS
// ═══════════════════════════════════════════════════
const Card = ({ children, style: extra }) => <div style={{ ...s.card, ...extra }}>{children}</div>;

const MetricCard = ({ label, value, sub, color }) => (
    <Card style={{ textAlign: 'center', flex: '1 1 140px', minWidth: 140 }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: color || '#e2e8f0', lineHeight: 1.2 }}>{value}</div>
        <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 0.5, marginTop: 4 }}>{label}</div>
        {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </Card>
);

const Empty = ({ msg }) => <Card style={{ textAlign: 'center', padding: 32, color: '#64748b', fontSize: 14 }}>{msg || 'No data available. Run analytics aggregation first.'}</Card>;

const TT = { backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 6, fontSize: 12 };

const DataTable = ({ columns, rows }) => (
    <div style={{ overflowX: 'auto', maxHeight: 360 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
                <tr>{columns.map((c, i) => <th key={i} style={s.th}>{c.label}</th>)}</tr>
            </thead>
            <tbody>
                {rows.map((row, ri) => (
                    <tr key={ri} style={{ backgroundColor: ri % 2 === 0 ? '#1a1f2e' : '#151922' }}>
                        {columns.map((c, ci) => <td key={ci} style={s.td}>{row[c.key] ?? '—'}</td>)}
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
);

// ═══════════════════════════════════════════════════
// TAB: KPI OVERVIEW
// ═══════════════════════════════════════════════════
const KPITab = ({ token }) => {
    const [d, setD] = useState(null);
    useEffect(() => { api('/kpis', token).then(r => setD(r.data)).catch(() => { }); }, [token]);
    if (!d?.has_data) return <Empty />;
    const k = d.kpis;
    const rc = d.raw_event_counts || {};
    return (<>
        {d.data_source && <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>Data source: {d.data_source}</div>}
        <div style={s.row}><MetricCard label="Searches" value={k.searches} color={C.primary} /><MetricCard label="Detections" value={k.detections} color={C.secondary} /><MetricCard label="Clicks" value={k.clicks} color={C.accent} /><MetricCard label="Buy Clicks" value={k.buys} color={C.success} /></div>
        <div style={s.row}><MetricCard label="Detection Success" value={`${k.detection_success}%`} /><MetricCard label="Search → Click" value={`${k.search_click_rate}%`} /><MetricCard label="Top-1 Click Rate" value={`${k.top1_rate}%`} /><MetricCard label="Conversion Rate" value={`${k.conversion_rate}%`} color={C.success} /></div>
        <div style={s.row}><MetricCard label="No-Click Rate" value={`${k.no_click_rate}%`} sub={k.no_click_rate > 40 ? '⚠ High' : 'OK'} color={k.no_click_rate > 40 ? C.danger : undefined} /><MetricCard label="Search Failure" value={`${k.search_fail_rate}%`} /><MetricCard label="Override Rate" value={`${k.override_rate}%`} /><MetricCard label="Relaxation Rate" value={`${k.relaxation_rate}%`} /></div>
        <Card>
            <h3 style={s.title}>Conversion Funnel</h3>
            <ResponsiveContainer width="100%" height={280}>
                <BarChart data={d.funnel}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="stage" stroke="#94a3b8" fontSize={12} /><YAxis stroke="#94a3b8" fontSize={12} /><Tooltip contentStyle={TT} /><Bar dataKey="count" radius={[4, 4, 0, 0]}>{d.funnel.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}</Bar></BarChart>
            </ResponsiveContainer>
        </Card>
        {Object.keys(rc).length > 0 && <Card>
            <h3 style={s.title}>Raw Event Counts (Real-time from DB)</h3>
            <p style={s.caption}>Direct count from analytics_events table — updates with every user action</p>
            <div style={s.row}>
                {Object.entries(rc).sort((a, b) => b[1] - a[1]).map(([type, count]) => (
                    <MetricCard key={type} label={type.replace(/_/g, ' ')} value={count} />
                ))}
            </div>
        </Card>}
    </>);
};

// ═══════════════════════════════════════════════════
// TAB: SYSTEM HEALTH
// ═══════════════════════════════════════════════════
const HealthTab = ({ token }) => {
    const [d, setD] = useState(null);
    useEffect(() => { api('/system-health', token).then(r => setD(r.data)).catch(() => { }); }, [token]);
    if (!d?.has_data) return <Empty />;
    return (<>
        <div style={s.row}>
            <MetricCard label="YOLO Confidence" value={`${d.yolo_confidence}%`} color={C.primary} />
            <MetricCard label="Search Failure" value={`${d.search_fail_rate}%`} sub={d.search_fail_rate > 25 ? '⚠ Weak catalog' : 'OK'} />
            <MetricCard label="Override Rate" value={`${d.override_rate}%`} sub={d.override_rate > 30 ? '⚠ AGMAN weak' : 'OK'} />
            <MetricCard label="Relaxation Rate" value={`${d.relaxation_rate}%`} />
        </div>
        <div style={s.row}>
            <MetricCard label="Avg Latency" value={`${d.latency.avg}ms`} />
            <MetricCard label="p50 (Median)" value={`${d.latency.p50}ms`} />
            <MetricCard label="p95" value={`${d.latency.p95}ms`} color={d.latency.p95 > 5000 ? C.danger : undefined} />
        </div>
        {d.latency_trend.length > 1 && <Card>
            <h3 style={s.title}>Latency Trend</h3>
            <ResponsiveContainer width="100%" height={260}>
                <LineChart data={d.latency_trend}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} /><Tooltip contentStyle={TT} /><Legend wrapperStyle={{ fontSize: 12 }} /><Line type="monotone" dataKey="avg" stroke={C.primary} strokeWidth={2} dot={false} /><Line type="monotone" dataKey="p50" stroke={C.secondary} strokeWidth={2} dot={false} /><Line type="monotone" dataKey="p95" stroke={C.danger} strokeWidth={2} dot={false} /></LineChart>
            </ResponsiveContainer>
        </Card>}
        {d.detection_categories.length > 0 && <Card>
            <h3 style={s.title}>Detection Analysis by Category</h3>
            <p style={s.caption}>YOLO performance per object type — low confidence = model weakness</p>
            <DataTable columns={[{ key: 'category', label: 'Category' }, { key: 'detections', label: 'Detections' }, { key: 'avg_confidence', label: 'Avg Confidence' }, { key: 'pct_of_total', label: '% of Total' }]} rows={d.detection_categories.map(c => ({ ...c, avg_confidence: `${c.avg_confidence}%`, pct_of_total: `${c.pct_of_total}%` }))} />
            {d.detection_summary.total_frames > 0 && <div style={{ ...s.row, marginTop: 12 }}>
                <MetricCard label="Total Frames" value={d.detection_summary.total_frames} />
                <MetricCard label="With Detection" value={d.detection_summary.with_detection} color={C.success} />
                <MetricCard label="No Detection" value={d.detection_summary.no_detection} color={C.danger} />
            </div>}
        </Card>}
    </>);
};

// ═══════════════════════════════════════════════════
// TAB: RECOMMENDATION QUALITY
// ═══════════════════════════════════════════════════
const QualityTab = ({ token }) => {
    const [d, setD] = useState(null);
    useEffect(() => { api('/recommendation-quality', token).then(r => setD(r.data)).catch(() => { }); }, [token]);
    if (!d?.has_data) return <Empty />;
    return (<>
        <div style={s.grid2}>
            <Card>
                <h3 style={s.title}>CTR by Rank Position</h3>
                <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={d.ctr_by_rank}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="rank" stroke="#94a3b8" fontSize={12} /><YAxis stroke="#94a3b8" fontSize={12} unit="%" /><Tooltip contentStyle={TT} /><Bar dataKey="ctr" fill={C.primary} radius={[4, 4, 0, 0]} /></BarChart>
                </ResponsiveContainer>
            </Card>
            <Card>
                <h3 style={s.title}>Top-1 Success Rate Trend</h3>
                <p style={s.caption}>Higher = retrieval ranks best product first</p>
                <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={d.top1_trend}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} unit="%" /><Tooltip contentStyle={TT} /><Line type="monotone" dataKey="rate" stroke={C.success} strokeWidth={2} /></LineChart>
                </ResponsiveContainer>
            </Card>
        </div>
        <div style={s.row}><MetricCard label="Avg Visual Similarity (Clicked)" value={d.avg_visual_similarity} color={C.secondary} /></div>
    </>);
};

// ═══════════════════════════════════════════════════
// TAB: BUSINESS INSIGHTS
// ═══════════════════════════════════════════════════
const BusinessTab = ({ token }) => {
    const [d, setD] = useState(null);
    useEffect(() => { api('/business-insights', token).then(r => setD(r.data)).catch(() => { }); }, [token]);
    if (!d?.has_data) return <Empty />;
    return (<>
        <div style={s.grid3}>
            <Card>
                <h3 style={s.title}>Top Categories</h3>
                <ResponsiveContainer width="100%" height={260}>
                    <PieChart><Pie data={d.top_categories} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false} fontSize={11}>{d.top_categories.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}</Pie><Tooltip contentStyle={TT} /></PieChart>
                </ResponsiveContainer>
            </Card>
            <Card>
                <h3 style={s.title}>Top Colors</h3>
                <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={d.top_colors} layout="vertical" margin={{ left: 60 }}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis type="number" stroke="#94a3b8" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} /><Tooltip contentStyle={TT} /><Bar dataKey="value" fill={C.accent} radius={[0, 4, 4, 0]} /></BarChart>
                </ResponsiveContainer>
            </Card>
            <Card>
                <h3 style={s.title}>Scene Distribution</h3>
                <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={d.scene_distribution} layout="vertical" margin={{ left: 80 }}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis type="number" stroke="#94a3b8" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={10} /><Tooltip contentStyle={TT} /><Bar dataKey="value" fill={C.secondary} radius={[0, 4, 4, 0]} /></BarChart>
                </ResponsiveContainer>
            </Card>
        </div>
        {d.scene_click_correlation.length > 0 && <Card>
            <h3 style={s.title}>Scene → Category Click Correlation</h3>
            <DataTable columns={[{ key: 'scene', label: 'Scene' }, { key: 'category', label: 'Category' }, { key: 'clicks', label: 'Clicks' }]} rows={d.scene_click_correlation} />
        </Card>}
        {d.demand_supply_gap.length > 0 && <Card>
            <h3 style={s.title}>Demand vs Supply Gap</h3>
            <p style={s.caption}>gap_score = (demand / catalog_size) × 100. High = inventory shortage.</p>
            <DataTable columns={[{ key: 'category', label: 'Category' }, { key: 'demand', label: 'Demand' }, { key: 'catalog_size', label: 'Catalog' }, { key: 'avg_results', label: 'Avg Results' }, { key: 'gap_score', label: 'Gap Score' }]} rows={d.demand_supply_gap} />
        </Card>}
    </>);
};

// ═══════════════════════════════════════════════════
// TAB: USER BEHAVIOUR
// ═══════════════════════════════════════════════════
const BehaviourTab = ({ token }) => {
    const [d, setD] = useState(null);
    useEffect(() => { api('/user-behaviour', token).then(r => setD(r.data)).catch(() => { }); }, [token]);
    if (!d?.has_data) return <Empty />;
    return (<>
        <div style={s.row}>
            <MetricCard label="Avg Searches / Session" value={d.avg_searches_per_session} color={C.primary} />
            <MetricCard label="Avg Filters / Search" value={d.avg_filters_per_search} color={C.secondary} />
            <MetricCard label="Impressions" value={d.impressions} />
        </div>
        {d.top_filters.length > 0 && <Card>
            <h3 style={s.title}>Most Used Filters</h3>
            <ResponsiveContainer width="100%" height={260}>
                <BarChart data={d.top_filters}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="name" stroke="#94a3b8" fontSize={12} /><YAxis stroke="#94a3b8" fontSize={12} /><Tooltip contentStyle={TT} /><Bar dataKey="value" fill={C.primary} radius={[4, 4, 0, 0]} /></BarChart>
            </ResponsiveContainer>
        </Card>}
        {d.session_trend.length > 1 && <Card>
            <h3 style={s.title}>Session Depth Trend</h3>
            <ResponsiveContainer width="100%" height={240}>
                <LineChart data={d.session_trend}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} /><Tooltip contentStyle={TT} /><Legend wrapperStyle={{ fontSize: 12 }} /><Line type="monotone" dataKey="searches_per_session" stroke={C.primary} strokeWidth={2} name="Searches/Session" /><Line type="monotone" dataKey="filters_per_search" stroke={C.accent} strokeWidth={2} name="Filters/Search" /></LineChart>
            </ResponsiveContainer>
        </Card>}
    </>);
};

// ═══════════════════════════════════════════════════
// TAB: DAILY TRENDS
// ═══════════════════════════════════════════════════
const TrendsTab = ({ token }) => {
    const [d, setD] = useState(null);
    useEffect(() => { api('/trends', token).then(r => setD(r.data)).catch(() => { }); }, [token]);
    if (!d?.has_data) return <Empty />;
    return (<>
        <div style={s.grid2}>
            <Card>
                <h3 style={s.title}>Volume</h3>
                <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={d.volume}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} /><Tooltip contentStyle={TT} /><Legend wrapperStyle={{ fontSize: 12 }} /><Line type="monotone" dataKey="searches" stroke={C.primary} strokeWidth={2} name="Searches" /><Line type="monotone" dataKey="clicks" stroke={C.accent} strokeWidth={2} name="Clicks" /><Line type="monotone" dataKey="buys" stroke={C.success} strokeWidth={2} name="Buys" /></LineChart>
                </ResponsiveContainer>
            </Card>
            <Card>
                <h3 style={s.title}>Latency</h3>
                <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={d.latency}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} unit="ms" /><Tooltip contentStyle={TT} /><Legend wrapperStyle={{ fontSize: 12 }} /><Line type="monotone" dataKey="avg" stroke={C.primary} strokeWidth={2} name="Avg" /><Line type="monotone" dataKey="p95" stroke={C.danger} strokeWidth={2} name="p95" /></LineChart>
                </ResponsiveContainer>
            </Card>
        </div>
    </>);
};

// ═══════════════════════════════════════════════════
// TAB: RAW EVENTS
// ═══════════════════════════════════════════════════
const EventsTab = ({ token }) => {
    const [d, setD] = useState(null);
    const [eventType, setEventType] = useState('All');
    const load = useCallback(t => { api('/raw-events', token, { params: { event_type: t } }).then(r => setD(r.data)).catch(() => { }); }, [token]);
    useEffect(() => { load(eventType); }, [eventType, load]);
    return (<Card>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <label style={{ color: '#94a3b8', fontSize: 13 }}>Event Type:</label>
            <select value={eventType} onChange={e => setEventType(e.target.value)} style={s.select}>
                {(d?.event_types || ['All']).map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <span style={{ color: '#64748b', fontSize: 12 }}>{d?.count || 0} events</span>
        </div>
        {d?.events?.length > 0
            ? <DataTable columns={[{ key: 'event_type', label: 'Type' }, { key: 'detected_category', label: 'Category' }, { key: 'detected_color', label: 'Color' }, { key: 'scene_label', label: 'Scene' }, { key: 'rank_clicked', label: 'Rank' }, { key: 'latency_ms', label: 'Latency' }, { key: 'created_at', label: 'Time' }]} rows={d.events} />
            : <div style={{ color: '#64748b', padding: 20, textAlign: 'center' }}>No events found</div>
        }
    </Card>);
};

// ═══════════════════════════════════════════════════
// TAB: LLM INSIGHTS
// ═══════════════════════════════════════════════════
const InsightsTab = ({ token }) => {
    const [d, setD] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [busy, setBusy] = useState(false);
    const [expanded, setExpanded] = useState(null);

    useEffect(() => { api('/ratings', token).then(r => setD(r.data)).catch(() => { }); }, [token]);

    const analyze = async () => {
        if (!d?.ratings?.length) return;
        setBusy(true);
        try {
            const res = await api('/insights', token, { method: 'POST', body: { rating_data: d.ratings[0] } });
            setAnalysis(res.data.analysis);
        } catch (e) { setAnalysis({ error: e.message || 'Failed' }); }
        setBusy(false);
    };

    if (!d || d.count === 0) return <Empty msg="No ratings data. Users need to submit ratings from the product page." />;
    const su = d.summary;
    const bc = { High: '#10b981', Medium: '#f59e0b', Low: '#ef4444' };

    return (<>
        <div style={s.row}>
            <MetricCard label="Avg Rating" value={`${su.avg_rating} ⭐`} color={C.accent} />
            <MetricCard label="Total Sessions" value={su.total_sessions} color={C.primary} />
            <MetricCard label="High Satisfaction (≥4)" value={su.high_satisfaction} color={C.success} />
        </div>
        <Card>
            <button onClick={analyze} disabled={busy} style={s.primaryBtn}>{busy ? '⏳ Generating...' : '🤖 Generate LLM Key Findings'}</button>
            {analysis && !analysis.error && <div style={{ marginTop: 20 }}>
                <h3 style={s.title}>Key Findings</h3>
                <div style={{ ...s.row, marginBottom: 16 }}>
                    <span style={{ ...s.badge, backgroundColor: bc[analysis.relevance_level] || '#64748b' }}>Relevance: {analysis.relevance_level}</span>
                    <span style={{ ...s.badge, backgroundColor: analysis.successful_recommendation ? '#10b981' : '#ef4444' }}>{analysis.successful_recommendation ? '✅ Successful' : '❌ Unsuccessful'}</span>
                </div>
                <div style={s.grid2}>
                    <div><h4 style={{ fontSize: 14, fontWeight: 600, color: C.success, marginBottom: 8, marginTop: 0 }}>💪 Strengths</h4>{(analysis.strengths || []).map((x, i) => <div key={i} style={{ fontSize: 13, color: '#cbd5e1', marginBottom: 4, paddingLeft: 8 }}>• {x}</div>)}</div>
                    <div><h4 style={{ fontSize: 14, fontWeight: 600, color: C.danger, marginBottom: 8, marginTop: 0 }}>⚠️ Weaknesses</h4>{(analysis.weaknesses || []).map((x, i) => <div key={i} style={{ fontSize: 13, color: '#fca5a5', marginBottom: 4, paddingLeft: 8 }}>• {x}</div>)}</div>
                </div>
                {analysis.improvement_suggestion && <div style={s.suggestion}><strong>💡 Technical Improvement:</strong> {analysis.improvement_suggestion}</div>}
                {analysis.user_behavior_analysis && <div style={{ marginTop: 12, color: '#cbd5e1', fontSize: 13 }}><strong>👤 User Behavior:</strong> {analysis.user_behavior_analysis}</div>}
            </div>}
            {analysis?.error && <div style={{ color: '#fca5a5', marginTop: 12 }}>Error: {analysis.error}</div>}
        </Card>
        <Card>
            <h3 style={s.title}>Recent Rating Entries</h3>
            {d.ratings.map(r => (
                <div key={r.id} style={{ borderBottom: '1px solid #1e293b', padding: '10px 0', cursor: 'pointer' }} onClick={() => setExpanded(expanded === r.id ? null : r.id)}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, color: '#e2e8f0' }}>
                        <span>Session #{r.id}</span>
                        <span>{'⭐'.repeat(r.rating || 0)} ({r.rating}/5)</span>
                        <span style={{ color: '#64748b' }}>Product: {r.product_id || '?'}</span>
                        <span style={{ color: '#475569', fontSize: 11 }}>{expanded === r.id ? '▲' : '▼'}</span>
                    </div>
                    {expanded === r.id && <pre style={{ backgroundColor: '#0f1117', padding: 12, borderRadius: 6, fontSize: 11, color: '#94a3b8', marginTop: 8, overflow: 'auto', maxHeight: 200 }}>{JSON.stringify(r, null, 2)}</pre>}
                </div>
            ))}
        </Card>
    </>);
};

// ═══════════════════════════════════════════════════
// MAIN DASHBOARD WITH TAB NAVIGATION
// ═══════════════════════════════════════════════════
const AdminDashboard = () => {
    const [token, setToken] = useState(() => sessionStorage.getItem('admin_token'));
    const [activeTab, setActiveTab] = useState('kpis');
    const [refreshing, setRefreshing] = useState(false);
    const [refreshKey, setRefreshKey] = useState(0);

    const handleLogin = t => { sessionStorage.setItem('admin_token', t); setToken(t); };
    const handleLogout = () => { sessionStorage.removeItem('admin_token'); setToken(null); };

    const handleRefresh = async () => {
        setRefreshing(true);
        try { await api('/refresh', token, { method: 'POST' }); } catch { }
        setRefreshKey(k => k + 1);
        setRefreshing(false);
    };

    if (!token) return <LoginScreen onLogin={handleLogin} />;

    const TabContent = { kpis: KPITab, health: HealthTab, quality: QualityTab, business: BusinessTab, behaviour: BehaviourTab, trends: TrendsTab, events: EventsTab, insights: InsightsTab };
    const ActiveComponent = TabContent[activeTab];

    return (
        <div style={s.page}>
            {/* HEADER */}
            <header style={s.header}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
                    <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0, color: '#f1f5f9' }}>📊 ShopWhatYouSee</h1>
                    <span style={{ fontSize: 13, color: '#64748b' }}>Commerce Intelligence Dashboard</span>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button onClick={handleRefresh} disabled={refreshing} style={s.refreshBtn}>
                        {refreshing ? '⏳ Refreshing...' : '🔄 Refresh Data'}
                    </button>
                    <button onClick={handleLogout} style={s.logoutBtn}>Logout</button>
                </div>
            </header>

            <div style={s.layout}>
                {/* SIDEBAR NAV */}
                <nav style={s.sidebar}>
                    {TABS.map(tab => (
                        <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                            style={activeTab === tab.id ? { ...s.tabBtn, ...s.tabActive } : s.tabBtn}>
                            <span style={{ fontSize: 16 }}>{tab.icon}</span>
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </nav>

                {/* MAIN CONTENT */}
                <main style={s.main}>
                    <h2 style={{ fontSize: 20, fontWeight: 600, color: '#f1f5f9', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
                        {TABS.find(t => t.id === activeTab)?.icon} {TABS.find(t => t.id === activeTab)?.label}
                    </h2>
                    <ActiveComponent key={`${activeTab}-${refreshKey}`} token={token} />
                </main>
            </div>
        </div>
    );
};

// ═══════════════════════════════════════════════════
// STYLES
// ═══════════════════════════════════════════════════
const s = {
    page: { backgroundColor: '#0f1117', minHeight: '100vh', color: '#e2e8f0', fontFamily: "'Inter', -apple-system, sans-serif" },
    header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 24px', borderBottom: '1px solid #1e293b', backgroundColor: '#0f1117', position: 'sticky', top: 0, zIndex: 100 },
    layout: { display: 'flex', minHeight: 'calc(100vh - 50px)' },
    sidebar: { width: 200, borderRight: '1px solid #1e293b', padding: '12px 0', display: 'flex', flexDirection: 'column', gap: 2, backgroundColor: '#0f1117', position: 'sticky', top: 50, height: 'calc(100vh - 50px)', overflowY: 'auto' },
    tabBtn: { display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', border: 'none', background: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 13, textAlign: 'left', width: '100%', borderRadius: 0, borderLeft: '3px solid transparent', transition: 'all 0.15s' },
    tabActive: { color: '#e2e8f0', backgroundColor: '#1a1f2e', borderLeftColor: '#6366f1', fontWeight: 600 },
    main: { flex: 1, padding: '24px 32px', maxWidth: 1100, display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto' },
    card: { backgroundColor: '#1a1f2e', borderRadius: 8, padding: 20, border: '1px solid #1e293b' },
    row: { display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 4 },
    grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 },
    grid3: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 },
    title: { fontSize: 15, fontWeight: 600, color: '#e2e8f0', marginBottom: 12, marginTop: 0 },
    caption: { fontSize: 12, color: '#64748b', marginTop: -8, marginBottom: 12 },
    th: { textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid #334155', color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', position: 'sticky', top: 0, backgroundColor: '#1a1f2e' },
    td: { padding: '6px 12px', borderBottom: '1px solid #1e293b', color: '#cbd5e1' },
    select: { backgroundColor: '#1e293b', color: '#e2e8f0', border: '1px solid #334155', padding: '6px 12px', borderRadius: 6, fontSize: 13 },
    primaryBtn: { backgroundColor: '#6366f1', color: '#fff', border: 'none', padding: '10px 24px', borderRadius: 6, cursor: 'pointer', fontSize: 14, fontWeight: 600, width: '100%' },
    refreshBtn: { backgroundColor: '#1e293b', color: '#94a3b8', border: '1px solid #334155', padding: '6px 14px', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 500 },
    logoutBtn: { background: 'none', border: '1px solid #334155', color: '#94a3b8', padding: '6px 14px', borderRadius: 6, cursor: 'pointer', fontSize: 12 },
    badge: { padding: '4px 12px', borderRadius: 12, fontSize: 12, fontWeight: 600, color: '#fff', display: 'inline-block' },
    suggestion: { marginTop: 16, padding: 12, borderRadius: 6, backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: 13, color: '#93c5fd' },
    loginContainer: { display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', backgroundColor: '#0f1117' },
    loginBox: { backgroundColor: '#1a1f2e', padding: 40, borderRadius: 12, border: '1px solid #1e293b', width: 360, textAlign: 'center' },
    loginInput: { width: '100%', padding: '10px 14px', backgroundColor: '#0f1117', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', fontSize: 14, boxSizing: 'border-box', marginBottom: 12 },
};

export default AdminDashboard;
