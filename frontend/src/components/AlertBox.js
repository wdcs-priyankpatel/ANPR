// frontend/src/components/AlertBox.js

import React from 'react';
import './AlertBox.css';

function AlertBox({ message, type, onClose }) {
  const isShowButton = type !== 'loading';

  return (
    <div className="alert-overlay">
      <div className={`alert-box ${type}`}>
        {type === 'loading' && <div className="spinner"></div>}
        <p>{message}</p>
        {isShowButton && (
          <button className="alert-button" onClick={onClose}>
            OK
          </button>
        )}
      </div>
    </div>
  );
}

export default AlertBox;