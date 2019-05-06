import React from 'react';

const LeftView = () => {
  return (
    <React.Fragment>
      <div>
        <svg
          className="bd-placeholder-img bd-placeholder-img-lg featurette-image img-fluid mx-auto"
          width="600"
          height="450"
          xmlns="http://www.w3.org/2000/svg"
          preserveAspectRatio="xMidYMid slice"
          focusable="false"
          role="img"
          aria-label="Placeholder: 500x500"
        >
          <title>Placeholder</title>
          <rect width="100%" height="100%" fill="#eee" />
          <text x="50%" y="50%" fill="#aaa" dy=".3em">
            500x500
          </text>
        </svg>
      </div>
      <button type="button" className="btn btn-primary btn-lg  btn-warning">
        Download
      </button>
    </React.Fragment>
  );
};

export default LeftView;
