
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Browse from './pages/Browse';
import Watch from './pages/Watch';
import ProductPage from './pages/ProductPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Browse />} />
        <Route path="/watch/:id" element={<Watch />} />
        <Route path="/product/:productId" element={<ProductPage />} />
      </Routes>
    </Router>
  );
}

export default App;
