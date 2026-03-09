
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Browse from './pages/Browse';
import Watch from './pages/Watch';
import ProductPage from './pages/ProductPage';
import AdminDashboard from './pages/AdminDashboard';
import UserProfile from './pages/UserProfile';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Browse />} />
        <Route path="/watch/:id" element={<Watch />} />
        <Route path="/product/:productId" element={<ProductPage />} />
        <Route path="/admin" element={<AdminDashboard />} />
        <Route path="/profile" element={<UserProfile />} />

      </Routes>
    </Router>
  );
}

export default App;
