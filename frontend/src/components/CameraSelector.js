import React, { useState, useEffect, useRef } from 'react';
import './CameraSelector.css';

function CameraSelector({ cameras, selectedCamera, onCameraChange }) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);
  
  const allCameras = [{ name: 'All', id: 'all' }, ...cameras];

  const toggleDropdown = () => setIsOpen(!isOpen);

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [dropdownRef]);

  const handleCameraClick = (camera) => {
    onCameraChange(camera);
    setIsOpen(false);
  };
  
  useEffect(() => {
    if (!selectedCamera && allCameras.length > 0) {
      onCameraChange(allCameras[0]);
    }
  }, [cameras, selectedCamera, onCameraChange]);

  return (
    <div className="camera-selector" ref={dropdownRef}>
      <button className="dropdown-toggle" onClick={toggleDropdown}>
        {selectedCamera ? selectedCamera.name : 'All'}
        <span className="dropdown-arrow">▼</span>
      </button>
      {isOpen && (
        <ul className="dropdown-menu">
          {allCameras.map(camera => (
            <li 
              key={camera.id} 
              className={`dropdown-item ${selectedCamera?.id === camera.id ? 'selected' : ''}`}
              onClick={() => handleCameraClick(camera)}
            >
              {camera.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default CameraSelector;