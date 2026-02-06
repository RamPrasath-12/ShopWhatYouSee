
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Browse from './pages/Browse';
import Watch from './pages/Watch';
import ProductPage from './pages/ProductPage';
import Insights from './pages/Insights';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Browse />} />
        <Route path="/watch/:id" element={<Watch />} />
        <Route path="/product/:productId" element={<ProductPage />} />
        <Route path="/insights" element={<Insights />} />
      </Routes>
    </Router>
  );
}

export default App;
