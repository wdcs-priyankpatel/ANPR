import React, { useState, useEffect } from 'react';
import './CameraSidebar.css';

function CameraSidebar({ 
  cameras, 
  onAddCamera, 
  onEditCamera,
  editingCamera,
  onSaveEdit,
  onCancelEdit,
  onToggleDetection
}) {
  const [form, setForm] = useState({ name: '', rtsp_link: '', coordinates: '' });
  const [editForm, setEditForm] = useState(editingCamera || { name: '', rtsp_link: '', coordinates: '' });
  const [isListOpen, setIsListOpen] = useState(false);
  const [startingCamera, setStartingCamera] = useState(null);

  useEffect(() => {
    if (editingCamera) {
      setEditForm(editingCamera);
    }
  }, [editingCamera]);

  const handleFormChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };
  
  const handleEditFormChange = (e) => {
    setEditForm({ ...editForm, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onAddCamera(form);
    setForm({ name: '', rtsp_link: '', coordinates: '' });
  };
  
  const handleSave = (e) => {
    e.preventDefault();
    onSaveEdit(editForm);
  };

  const toggleList = () => {
    setIsListOpen(!isListOpen);
  };
  
  const handleToggle = (camera, currentStatus) => {
    if (!camera || camera.id === 'all' || !camera.id) {
      alert("Please select a valid camera to start/stop detection.");
      return;
    }

    if (currentStatus === 'stopped') {
      setStartingCamera(camera.id);
    } else {
      setStartingCamera(null);
    }
    onToggleDetection(camera.id, currentStatus); // CHANGED from `(camera, currentStatus)`
  };

  return (
    <div className="sidebar">
      <h2 onClick={toggleList} className="cameras-dropdown-header">
        Cameras <span className="dropdown-arrow">{isListOpen ? '▲' : '▼'}</span>
      </h2>
      
      {isListOpen && (
        <ul className="camera-list">
          {cameras.map(camera => (
            <li key={camera.id}>
              <div className="camera-info">
                {camera.name}
              </div>
              <div className="camera-actions">
                <button
                  className="edit-button"
                  onClick={() => onEditCamera(camera)}
                >
                  Edit
                </button>
                <button
                  className={`detection-button ${camera.status}`}
                  onClick={() => handleToggle(camera, camera.status)}
                  disabled={startingCamera === camera.id}
                >
                  {startingCamera === camera.id ? 'Starting...' : (camera.status === 'running' ? 'Stop' : 'Start')}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      
      {editingCamera ? (
        <div className="add-camera-form">
          <h3>Edit Camera</h3>
          <form onSubmit={handleSave}>
            <input
              type="text"
              name="name"
              value={editForm.name}
              onChange={handleEditFormChange}
              placeholder="Camera Name"
              required
            />
            <input
              type="text"
              name="rtsp_link"
              value={editForm.rtsp_link}
              onChange={handleEditFormChange}
              placeholder="RTSP Link"
              required
            />
            <input
              type="text"
              name="coordinates"
              value={editForm.coordinates}
              onChange={handleEditFormChange}
              placeholder="Coordinates (x1,y1,x2,y2)"
              required
            />
            <div className="form-actions">
              <button type="submit" className="save-button">Save</button>
              <button type="button" onClick={onCancelEdit} className="cancel-button">✖ Cancel</button>
            </div>
          </form>
        </div>
      ) : (
        <div className="add-camera-form">
          <h3>Add New Camera</h3>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              name="name"
              value={form.name}
              onChange={handleFormChange}
              placeholder="Camera Name"
              required
            />
            <input
              type="text"
              name="rtsp_link"
              value={form.rtsp_link}
              onChange={handleFormChange}
              placeholder="RTSP Link"
              required
            />
            <input
              type="text"
              name="coordinates"
              value={form.coordinates}
              onChange={handleFormChange}
              placeholder="Coordinates (x1,y1,x2,y2)"
              required
            />
            <button type="submit">Add Camera</button>
          </form>
        </div>
      )}
    </div>
  );
}

export default CameraSidebar;