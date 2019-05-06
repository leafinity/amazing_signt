import React, { Component } from 'react';
import Pagination from './paginate';
import Generate from './generate';

class RightView extends Component {
  // you can put all the value inside the state. don't assign controller's state
  // custom style writes here
  state = {
    currentPage: 'Mountain',
    photoType: ['Mountain', 'Desert', 'Castle', 'Waterfall']
  };
  handlePhotoChange = photo => {
    this.setState({ currentPage: photo });
  };

  handleGenerate = phtoo => {};

  render() {
    const { currentPage, photoType } = this.state;
    return (
      <React.Fragment>
        <Pagination
          currentPage={currentPage}
          photoType={photoType}
          onPhotoChange={this.handlePhotoChange}
        />
        <main className="container">
          <Generate OnGenerate={this.handleGenerate} />
          <div>
            <h4>Click Generate to create GAN image.</h4>
          </div>
        </main>
      </React.Fragment>
    );
  }
}

export default RightView;
