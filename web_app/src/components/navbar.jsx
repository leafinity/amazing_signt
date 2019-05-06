import React from 'react';

// stateless functional component sfc

const NavBar = () => {
  return (
    <nav className="navbar navbar-light bg-light" bg="secondary">
      <div
        className="navbar-brand"
        style={{
          fontSize: 25
        }}
      >
        Amazing Si
        <a style={{ color: 'red', fontSize: 25 }}>gn</a>h {''}
        <span className="badge-pill badge-warning">
          <span
            style={{
              fontSize: 15
            }}
          >
            w/
          </span>
          Tensorflow 2.0
        </span>
      </div>
      <span
        className="navbar-text"
        style={{
          float: 'right',
          color: '#000000'
        }}
      >
        English/Traditional Mandain
      </span>
    </nav>
  );
};

export default NavBar;
