import React from 'react';
import './NumberPlateTable.css';

function NumberPlateTable({ plates, onRowClick }) {
  return (
    <div className="table-container">
      <table className="number-plate-table">
        <thead>
          <tr>
            <th>Camera Name</th>
            <th>Plate Number</th>
            <th>Status</th> {/* NEW COLUMN: Status */}
            <th>Detection Time</th>
            <th>View Image</th>
          </tr>
        </thead>
        <tbody>
          {plates.length === 0 ? (
            <tr>
              {/* Updated colspan from 4 to 5 */}
              <td colSpan="5" className="no-plates-message">No number plates detected yet or matching your filters.</td>
            </tr>
          ) : (
            plates.map((plate) => (
              <tr key={plate.id}>
                <td>{plate.camera_name}</td>
                <td>{plate.plate_number}</td>
                {/* Dynamically apply class for status styling */}
                <td className={`status-${plate.status.toLowerCase()}`}>{plate.status}</td>
                <td>{new Date(plate.created_at).toLocaleString()}</td>
                <td>
                  <button
                    className="view-image-button"
                    onClick={() => onRowClick(plate.image_path)}
                    title="View Image"
                  >
                    👁️
                  </button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default NumberPlateTable;
