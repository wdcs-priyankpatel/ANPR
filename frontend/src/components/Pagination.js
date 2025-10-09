import React from 'react';
import './Pagination.css';

const Pagination = ({ platesPerPage, totalPlates, paginate, currentPage }) => {
  const pageNumbers = [];
  const totalPages = Math.ceil(totalPlates / platesPerPage);

  // Logic to determine which page numbers to show, including ellipsis
  const getPageNumbers = () => {
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
      }
    } else {
      if (currentPage <= 4) {
        for (let i = 1; i <= 5; i++) {
          pageNumbers.push(i);
        }
        pageNumbers.push('...');
        pageNumbers.push(totalPages);
      } else if (currentPage > totalPages - 4) {
        pageNumbers.push(1);
        pageNumbers.push('...');
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pageNumbers.push(i);
        }
      } else {
        pageNumbers.push(1);
        pageNumbers.push('...');
        for (let i = currentPage - 1; i <= currentPage + 1; i++) {
          pageNumbers.push(i);
        }
        pageNumbers.push('...');
        pageNumbers.push(totalPages);
      }
    }
    return pageNumbers;
  };

  const pages = getPageNumbers();

  return (
    <nav className="pagination-nav">
      <ul className="pagination">
        <li className={`page-item ${currentPage === 1 ? 'disabled' : ''}`}>
          <a onClick={() => paginate(currentPage - 1)} href='#!' className="page-link">
            &lt; Previous
          </a>
        </li>
        {pages.map(number => (
          <li key={number} className={`page-item ${currentPage === number ? 'active' : ''} ${number === '...' ? 'ellipsis' : ''}`}>
            {number === '...' ? (
              <span className="page-link">...</span>
            ) : (
              <a onClick={() => paginate(number)} href='#!' className="page-link">
                {number}
              </a>
            )}
          </li>
        ))}
        <li className={`page-item ${currentPage === totalPages ? 'disabled' : ''}`}>
          <a onClick={() => paginate(currentPage + 1)} href='#!' className="page-link">
            Next &gt;
          </a>
        </li>
      </ul>
    </nav>
  );
};

export default Pagination;