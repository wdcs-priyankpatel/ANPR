import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './index.css';
import NumberPlateTable from './components/NumberPlateTable';
import CameraSidebar from './components/CameraSidebar';
import ImageViewer from './components/ImageViewer';
import CameraSelector from './components/CameraSelector';
import Pagination from './components/Pagination';

import 'react-date-range/dist/styles.css';
import 'react-date-range/dist/theme/default.css';
import { DateRange } from 'react-date-range';
import { addDays } from 'date-fns';
import { enUS } from 'date-fns/locale';

const API_URL = 'http://localhost:8000';
const PLATES_PER_PAGE = 20;

function App() {
  const [plates, setPlates] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState({ name: 'All', id: 'all' });
  const [selectedImage, setSelectedImage] = useState(null);
  const [showImage, setShowImage] = useState(false);
  const [editingCamera, setEditingCamera] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const [currentPage, setCurrentPage] = useState(1);
  const [currentPlates, setCurrentPlates] = useState([]);
  const [totalPlates, setTotalPlates] = useState(0);

  const [dateRange, setDateRange] = useState([
    {
      startDate: new Date(),
      endDate: addDays(new Date(), 0),
      key: 'selection'
    }
  ]);
  const [showCalendar, setShowCalendar] = useState(false);
  const [message, setMessage] = useState({ text: '', type: '' });
  const statusPollInterval = useRef(null);

  const fetchCameras = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/cameras`);
      setCameras(response.data);
    } catch (error) {
      console.error("Error fetching cameras:", error);
    }
  }, []);

  const fetchPlates = useCallback(async (cameraName = null, plateNumber = '', page = 1) => {
    try {
      const startDate = dateRange[0].startDate.toISOString().split('T')[0];
      const endDate = dateRange[0].endDate.toISOString().split('T')[0];

      const params = {
        ...(cameraName && cameraName !== 'All' && { camera_name: cameraName }),
        ...(plateNumber && { plate_number: plateNumber }),
        start_date: startDate,
        end_date: endDate,
        skip: (page - 1) * PLATES_PER_PAGE,
        limit: PLATES_PER_PAGE,
      };

      const response = await axios.get(`${API_URL}/number-plates`, { params });
      setPlates(response.data.data);
      setTotalPlates(response.data.total_count);
    } catch (error) {
      console.error("Error fetching number plates:", error);
    }
  }, [dateRange]);

  useEffect(() => {
    fetchCameras();
    fetchPlates();
  }, [fetchCameras, fetchPlates]);

  useEffect(() => {
    setCurrentPlates(plates);
  }, [plates]);

  const handleCameraChange = (camera) => {
    setSelectedCamera(camera);
    fetchPlates(camera.name, searchTerm, 1);
  };

  const handleFilterChange = () => {
    fetchPlates(selectedCamera?.name, searchTerm, 1);
    setShowCalendar(false);
  };

  const handleEditCamera = (camera) => {
    setEditingCamera(camera);
  };

  const handleSaveEdit = async (updatedCamera) => {
    try {
      await axios.put(`${API_URL}/cameras/${updatedCamera.id}`, updatedCamera);
      fetchCameras();
      setEditingCamera(null);
    } catch (error) {
      console.error("Error updating camera:", error);
      alert(`Failed to update camera: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleCancelEdit = () => {
    setEditingCamera(null);
  };

  const handleRowClick = (imagePath) => {
    setSelectedImage(`${API_URL}/images/${imagePath}`);
    setShowImage(true);
  };

  const handleAddCamera = async (newCamera) => {
    try {
      await axios.post(`${API_URL}/cameras`, newCamera);
      fetchCameras();
    } catch (error) {
      console.error("Error adding camera:", error);
      alert(`Failed to add camera: ${error.response?.data?.detail || error.message}`);
    }
  };

  const checkStatus = async (cameraId) => {
    try {
      const response = await axios.get(`${API_URL}/detection-status/${cameraId}`);
      if (response.data.status === 'running') {
        setMessage({ text: 'Detection started successfully.', type: 'success' });
        clearInterval(statusPollInterval.current);
        fetchCameras();
      } else if (response.data.status === 'error') {
        setMessage({ text: 'Error: Could not open video stream. Check your RTSP link.', type: 'error' });
        clearInterval(statusPollInterval.current);
        fetchCameras();
      }
    } catch (error) {
      setMessage({ text: 'Error checking detection status.', type: 'error' });
      clearInterval(statusPollInterval.current);
      fetchCameras();
    }
  };

  const handleToggleDetection = async (cameraId, currentStatus) => {
    try {
      const endpoint = currentStatus === 'running' ? 'stop-detection' : 'start-detection';
      const response = await axios.post(`${API_URL}/${endpoint}/${cameraId}`);

      if (response.data.status === 'starting') {
        setMessage({ text: `Attempting to start detection for Camera ${cameraId}...`, type: 'info' });
        statusPollInterval.current = setInterval(() => checkStatus(cameraId), 3000);
      } else if (response.data.status === 'stopped') {
        setMessage({ text: `Detection for Camera ${cameraId} stopped.`, type: 'success' });
        fetchCameras();
      } else {
        setMessage({ text: response.data.message, type: 'success' });
        fetchCameras();
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail
        ? typeof error.response.data.detail === 'string'
          ? error.response.data.detail
          : "An unknown error occurred."
        : error.message;

      setMessage({ text: `Failed to toggle detection: ${errorMessage}`, type: 'error' });
    }
  };

  const paginate = (pageNumber) => {
    setCurrentPage(pageNumber);
    fetchPlates(selectedCamera.name, searchTerm, pageNumber);
  };

  const handleRefresh = () => {
    fetchPlates(selectedCamera.name, searchTerm, currentPage);
  };

  return (
    <div className="app-container">
      <CameraSidebar
        cameras={cameras}
        onAddCamera={handleAddCamera}
        onEditCamera={handleEditCamera}
        editingCamera={editingCamera}
        onSaveEdit={handleSaveEdit}
        onCancelEdit={handleCancelEdit}
        onToggleDetection={handleToggleDetection}
      />
      <div className="main-content">
        <header className="main-header">
          <h1 className="main-title">
            Detected Number Plates {selectedCamera?.name && selectedCamera.name !== 'All' && `from ${selectedCamera.name}`}
          </h1>
          <div className="filters-container">
            <CameraSelector
              cameras={cameras}
              selectedCamera={selectedCamera}
              onCameraChange={handleCameraChange}
            />
            <div className="filter-group-search">
              <input
                type="text"
                className="search-input"
                placeholder="Search plate number..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyUp={(e) => { if (e.key === 'Enter') handleFilterChange(); }}
              />
            </div>
            <div className="filter-group-date">
              <div onClick={() => setShowCalendar(!showCalendar)} className="date-input-display">
                <span className="date-range-text">
                  {`${dateRange[0].startDate.toLocaleDateString()} - ${dateRange[0].endDate.toLocaleDateString()}`}
                </span>
              </div>
              {showCalendar && (
                <div className="calendar-popup">
                  <DateRange
                    editableDateInputs={true}
                    onChange={item => setDateRange([item.selection])}
                    moveRangeOnFirstSelection={false}
                    ranges={dateRange}
                    locale={enUS}
                  />
                  <button className="apply-filter-button" onClick={handleFilterChange}>Apply</button>
                </div>
              )}
            </div>
            <button className="refresh-button" onClick={handleRefresh}>🔄 Refresh</button>
          </div>
        </header>
        {message.text && (
          <div className={`message-box ${message.type}`}>
            {message.text}
          </div>
        )}
        <NumberPlateTable plates={currentPlates} onRowClick={handleRowClick} />
        {showImage && selectedImage && (
          <ImageViewer imageUrl={selectedImage} onClose={() => setShowImage(false)} />
        )}
        {totalPlates > PLATES_PER_PAGE && (
          <Pagination
            platesPerPage={PLATES_PER_PAGE}
            totalPlates={totalPlates}
            paginate={paginate}
            currentPage={currentPage}
          />
        )}
      </div>
    </div>
  );
}

export default App;