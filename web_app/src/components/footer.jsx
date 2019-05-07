import React from 'react';

const Footer = () => {
  const styles = {
    backgroundColor: '#F8F8F8',
    borderTop: '1px solid #E7E7E7',
    textAlign: 'center',
    padding: '20px',
    position: 'fixed',
    left: '0',
    bottom: '0',
    height: '50px',
    width: '100%'
  };
  return (
    <footer className="footer mt-auto py-3" style={styles}>
      <div className="container">
        <p className="mb-1">&copy; 2019 Abby & Abby's fellows</p>
      </div>
    </footer>
  );
};

export default Footer;
