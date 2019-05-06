import React from 'react';
import castle from '../img/castle.jpg';
import mountain from '../img/mountain.jpeg';
import desert from '../img/desert.jpg';
import waterfall from '../img/waterfall.jpeg';

const Pagination = props => {
  const { photoType, currentPage, onPhotoChange } = props;

  const textAlign = {
    textAlign: 'center'
  };
  const imgSize = {
    width: 10,
    height: 10
  };

  // show pics 接 server
  const showPic = pic => {
    if (pic === 'Mountain') return mountain;
    if (pic === 'Desert') return desert;
    if (pic === 'Castle') return castle;
    if (pic === 'Waterfall') return waterfall;
  };

  return (
    <React.Fragment>
      <nav className="row justify-content-around">
        <ul className="pagination">
          {photoType.map(photo => (
            <li
              key={photo}
              className={
                photo === currentPage ? 'page-item active' : 'page-item'
              }
            >
              <a className="page-link" onClick={() => onPhotoChange(photo)}>
                {photo}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      <p>choose the type of image you gonna make!</p>
      <div className="row justify-content-around">
        <img
          src={showPic(currentPage)}
          className="img-thumbnail"
          alt="logo"
          styles={imgSize}
          width="300"
          height="300"
        />
      </div>
      <p style={textAlign}>original Pics</p>
    </React.Fragment>
  );
};

export default Pagination;

// photoType.map(type => (     <div className="btn-group btn-group-lg"
// role="group" aria-label={type}>         <button type="button" className="btn
// btn-primary">{type}</button>     </div> ))
