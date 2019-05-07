import React, { Component } from 'react';
import Pagination from './paginate';
import axios from 'axios';
import file1 from '../img/demo/file1.png';

class TopView extends Component {
  // you can put all the value inside the state. don't assign controller's state
  // custom style writes here

  state = {
    link: file1,
    currentPage: 'Mountain',
    photoType: ['Mountain', 'Desert', 'Lake', 'Waterfall']
  };

  handlePhotoChange = photo => {
    this.setState({ currentPage: photo });
  };

  onGenerateChange = currentPage => {
    axios({
      url:
        'http://e20e6f3d.ngrok.io/generate_scene/' + currentPage.toLowerCase(),
      method: 'POST',
      responseType: 'blob' // important
    }).then(response => {
      const url = window.URL.createObjectURL(new Blob([response.data]));

      if (currentPage === 'Mountain') {
        this.setState({ link: url });
      } else if (currentPage === 'Desert') {
        this.setState({ link: url });
      } else if (currentPage === 'Lake') {
        this.setState({ link: url });
      } else if (currentPage === 'Waterfall') {
        this.setState({ link: url });
      }
    });
  };

  handleDownload = link => {
    const new_link = document.createElement('a');
    new_link.href = link;
    new_link.setAttribute('download', 'Amazing_signh.png'); //or any other extension
    document.body.appendChild(new_link);
    new_link.click();
  };

  render() {
    const { currentPage, photoType, link } = this.state;

    return (
      <React.Fragment>
        <div className="col-md-6 col justify-content-md-left">
          <Pagination
            currentPage={currentPage}
            photoType={photoType}
            onPhotoChange={this.handlePhotoChange}
          />
          <main className="container">
            <button
              type="button"
              className="btn btn-primary btn-lg btn-warning"
              onClick={() => this.onGenerateChange(currentPage)}
            >
              Generate
            </button>
            <div>
              <h4>Click Generate to create GAN image.</h4>
            </div>
          </main>
        </div>
        <div className="col-md-4 ">
          <div>
            <img
              src={link}
              style={{ width: 500, height: 500 }}
              alt="get nothing"
            />
            <title>Placeholder</title>
          </div>
          <div>
            <h3 />
            <button
              type="button"
              className="btn btn-primary btn-lg  btn-warning"
              onClick={() => this.handleDownload(link)}
            >
              Download
            </button>
          </div>
        </div>
      </React.Fragment>
    );
  }
}

export default TopView;
