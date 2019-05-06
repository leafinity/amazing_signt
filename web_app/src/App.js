import React, { Component } from 'react';
// import logo from './logo.svg';
import NavBar from './components/navbar';
import TopView from './components/TopView';
import './App.css';
import LowerView from './components/lowerView';
import Footer from './components/footer';

class App extends Component {
  state = {};
  render() {
    return (
      <React.Fragment>
        <NavBar />
        <div class="row">
          <div class="col-bg-4">
            <h3 />
          </div>
        </div>
        <section className=" text-center">
          <div className="row">
            <TopView />
          </div>
        </section>
        <LowerView />
        <Footer />
      </React.Fragment>
    );
  }
}

export default App;
