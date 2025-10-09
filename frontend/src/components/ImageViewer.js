import React from 'react';
import './ImageViewer.css';

function ImageViewer({ imageUrl, onClose }) {
  return (
    <div className="image-viewer-overlay" onClick={onClose}>
      <div className="image-viewer-content" onClick={(e) => e.stopPropagation()}>
        <span className="close-button" onClick={onClose}>&times;</span>
        <img src={imageUrl} alt="Detected Number Plate" />
      </div>
    </div>
  );
}

export default ImageViewer;