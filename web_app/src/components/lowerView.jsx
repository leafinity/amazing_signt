import React from 'react';
import githubMark from '../img/GitHub.png';
import demo1 from '../img/demo/demo1.jpeg';
import demo2 from '../img/demo/demo2.jpeg';
import demo3 from '../img/demo/demo3.jpeg';

const LowerView = () => {
  return (
    <React.Fragment>
      <div class="container marketing">
        <hr class="featurette-divider" />
        <hr class="featurette-divider" />
        <div class="row featurette">
          <div class="col-md-7">
            <h2 class="featurette-heading ">
              Ain't got no idea.
              <h3 class="text-muted"> Lemme got your back.</h3>
            </h2>
            <p class="lead">
              Think about it. If your were a game scene designer, and you run
              out of the creativities, but don't worry,
              <a style={{ color: 'orange' }}> Amazing Signt </a> will get your
              back. <a style={{ color: 'orange' }}> Amazing Signt </a> use SOTA
              deep learning mechanism to create the scene you have never seen in
              the world. Try to use
              <a style={{ color: 'orange' }}> Amazing Signt </a>to save your
              time. Now, you can use those created image as an reference to make
              an whole new landscape image on your jobs! Most important of all,
              pics are made by neural networks, you don't have to worry about
              the license issues.
            </p>
          </div>
          <div class="col-md-5">
            <img src={demo1} alt="Smiley face" height="300" width="400" />
            <title>Placeholder</title>
            <rect width="100%" height="100%" fill="#eee" />
            <text x="50%" y="50%" fill="#aaa" dy=".3em" />
          </div>
        </div>

        <hr class="featurette-divider" />
        <hr class="featurette-divider" />

        <div class="row featurette">
          <div class="col-md-7 order-md-2">
            <h2 class="featurette-heading">Ohhhh, that's dope!</h2>
            <h3 class="text-muted">
              Being the coolest guy in you friends zone.
            </h3>
            <p class="lead">
              Dunno what to post in your instagram? You must have to try
              <a style={{ color: 'orange' }}> Amazing Signt</a>. You can use
              <a style={{ color: 'orange' }}> Amazing Signt </a> APP to generate
              some awesome images and then posting on Instagram. Definitely, you
              are gonna be the coolest guy among your friends. It just like you
              are an legendary photographer traveling all around the world, and
              taking some amazing pictures.
            </p>
          </div>
          <div class="col-md-5 order-md-1">
            <img
              src={demo2}
              alt="Smiley face"
              height="300"
              width="400"
              style={{
                position: 'relative',
                left: '0px'
              }}
            />
            <title>Placeholder</title>
            <rect width="100%" height="100%" fill="#eee" />
            <text x="50%" y="50%" fill="#aaa" dy=".3em" />
          </div>
        </div>

        <hr className="featurette-divider" />
        <hr class="featurette-divider" />

        <div className="row featurette">
          <div className="col-md-7">
            <h2 className="featurette-heading">
              Trick your friends.
              <span className="text-muted"> Gotcha</span>
            </h2>
            <p className="lead">
              <a style={{ color: 'orange' }}> Amazing Signt </a> use Deep
              Learning to generate infinitely realistic pictures. you can show
              these pictures to your friends, and tell them you just go for a
              vacation and take so many beatiful pictures. Make up a ridiculous
              story and try to fool your friends! Finally, tell your friend that
              both the story and the photos are phony! Gotcha, bro!
            </p>
          </div>
          <div class="col-md-5">
            <img src={demo3} alt="Smiley face" height="300" width="400" />
            <title>Placeholder</title>
            <rect width="100%" height="100%" fill="#eee" />
            <text x="50%" y="50%" fill="#aaa" dy=".3em" />
          </div>
        </div>
        <hr className="featurette-divider" />
        <hr class="featurette-divider" />
        <div>
          <h2 className="text-mutedtext text-center">Demo Video</h2>
          <p align="center">
            {/* <iframe
              title="title"
              align="center"
              width="560"
              height="315"
              src="https://www.youtube.com/embed/videoseries?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG"
            /> */}
          </p>
        </div>
        <hr class="featurette-divider" />
        <div>
          <h2 class="text-mutedtext text-center">
            <a
              href="https://github.com/leafinity/amazing_signt"
              style={{ color: 'orange' }}
            >
              <h2 class="featurette-heading text-center">
                Learn more! Check out our Github
              </h2>
              <img src={githubMark} alt="Italian Trulli" />
            </a>
          </h2>

          <p class="lead" />
        </div>
        <div>
          <h2>c</h2>
        </div>
      </div>
    </React.Fragment>
  );
};
export default LowerView;
