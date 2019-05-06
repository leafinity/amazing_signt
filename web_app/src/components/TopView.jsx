import React, { Component } from 'react';
import Pagination from './paginate';
import axios from 'axios';
import file1 from '../img/demo/file1.png';
import file2 from '../img/demo/file2.png';
import file3 from '../img/demo/file3.png';
import file4 from '../img/demo/file4.png';
import file5 from '../img/demo/file5.png';
import file6 from '../img/demo/file6.png';
import file7 from '../img/demo/file7.png';
import file8 from '../img/demo/file8.png';
import file9 from '../img/demo/file9.png';
import file10 from '../img/demo/file10.png';
import file11 from '../img/demo/file11.png';
import file12 from '../img/demo/file12.png';
import { throwStatement } from '@babel/types';

class TopView extends Component {
  // you can put all the value inside the state. don't assign controller's state
  // custom style writes here
  moun = [file1, file2, file3, file4];
  des = [file5, file6, file7, file8];
  cas = [file9, file10, file11, file12];

  state = {
    link: file1,
    currentPage: 'Mountain',
    photoType: ['Mountain', 'Desert', 'Castle', 'Waterfall']
  };

  handlePhotoChange = photo => {
    this.setState({ currentPage: photo });
  };

  //   onGenerateChange = currentPage => {
  // axios({
  //   url: 'http://9bc25fba.ngrok.io/generate_scene/castle',
  //   method: 'POST',
  //   responseType: 'blob' // important
  // }).then(response => {
  //   const url = window.URL.createObjectURL(new Blob([response.data]));
  //   const link = document.createElement('a');
  //   link.href = url;
  //   link.setAttribute('download', 'file1.png'); //or any other extension
  //   document.body.appendChild(link);
  //   link.click();
  // });
  //   };
  onGenerateChange = currentPage => {
    const ran = Math.floor(Math.random() * 4);
    if (currentPage === 'Mountain') {
      this.setState({ link: this.moun[ran] });
    }
    if (currentPage === 'Desert') {
      this.setState({ link: this.des[ran] });
    }
  };

  render() {
    const { currentPage, photoType, link } = this.state;

    return (
      <React.Fragment>
        <div className="col-md-7 ">
          <div>
            <img
              src={link}
              style={{ width: 500, height: 500 }}
              alt="get nothing"
            />
            <title>Placeholder</title>
          </div>
          <button type="button" className="btn btn-primary btn-lg  btn-warning">
            Download
          </button>
        </div>
        <div className="col-md-4 ">
          <Pagination
            currentPage={currentPage}
            photoType={photoType}
            onPhotoChange={this.handlePhotoChange}
          />
          <main className="container">
            <button
              type="button"
              className="btn btn-primary btn-lg btn-block btn-warning"
              onClick={() => this.onGenerateChange(currentPage)}
            >
              Generate
            </button>
            <div>
              <h4>Click Generate to create GAN image.</h4>
            </div>
          </main>
        </div>
      </React.Fragment>
    );
  }
}

export default TopView;
