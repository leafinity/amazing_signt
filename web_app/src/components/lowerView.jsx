import React from 'react';
import githubMark from '../img/GitHub.png';
const LowerView = () => {
  return (
    <React.Fragment>
      <div class="container marketing">
        <hr class="featurette-divider" />
        <div class="row featurette">
          <div class="col-md-7">
            <h2 class="featurette-heading">
              Ain't got no idea.
              <h3 class="text-muted"> Lemme got your back.</h3>
            </h2>
            <p class="lead">
              Think about it. If your were a scence designer, and you run out of
              the creativities,
              <a style={{ color: 'orange' }}> Amazing Signt </a> will get your
              back. <a style={{ color: 'orange' }}> Amazing Signt </a> is gonna
              create a new scence you have never seen in the world. Try to use
              <a style={{ color: 'orange' }}> Amazing Signt </a>to save your
              time. Now, you can use the created image as an reference to make
              an whole new landscape image. Most important! No license issue!
            </p>
          </div>
          <div class="col-md-5">
            <svg
              class="bd-placeholder-img bd-placeholder-img-lg featurette-image img-fluid mx-auto"
              width="420"
              height="420"
              xmlns="http://www.w3.org/2000/svg"
              preserveAspectRatio="xMidYMid slice"
              focusable="false"
              role="img"
              aria-label="Placeholder: 420x420"
            >
              <title>Placeholder</title>
              <rect width="100%" height="100%" fill="#eee" />
              <text x="50%" y="50%" fill="#aaa" dy=".3em">
                500x500
              </text>
            </svg>
          </div>
        </div>

        <hr class="featurette-divider" />

        <div class="row featurette">
          <div class="col-md-7 order-md-2">
            <h2 class="featurette-heading">Ohhhh, that's dope!</h2>
            <h3 class="text-muted">Being the coolest guy in you zone.</h3>
            <p class="lead">
              Dunno what's to post in your instagram? Try to use
              <a style={{ color: 'orange' }}> Amazing Signt </a>
              to generate some images and posting in on Instagram. Definitely,
              you are going to be the coolest guy among your friends. Just like
              you are an awesome photographer traveling around the world, and
              shooting the amazing sight.
            </p>
          </div>
          <div class="col-md-5 order-md-1">
            <svg
              class="bd-placeholder-img bd-placeholder-img-lg featurette-image img-fluid mx-auto"
              width="450"
              height="450"
              xmlns="http://www.w3.org/2000/svg"
              preserveAspectRatio="xMidYMid slice"
              focusable="false"
              role="img"
              aria-label="Placeholder: 420x420"
            >
              <title>Placeholder</title>
              <rect width="100%" height="100%" fill="#eee" />
              <text x="50%" y="50%" fill="#aaa" dy=".3em">
                500x500
              </text>
            </svg>
          </div>
        </div>

        <hr class="featurette-divider" />

        <div class="row featurette">
          <div class="col-md-7">
            <h2 class="featurette-heading">
              Trick your friends.
              <span class="text-muted"> Gotcha</span>
            </h2>
            <p class="lead">
              Get the latest news and information for the Boston Celtics. 2018
              season schedule, scores, stats, and highlights. Find out the
              latest on your favorite NBA teams on ...
            </p>
          </div>
          <div class="col-md-5">
            <svg
              class="bd-placeholder-img bd-placeholder-img-lg featurette-image img-fluid mx-auto"
              width="420"
              height="420"
              xmlns="http://www.w3.org/2000/svg"
              preserveAspectRatio="xMidYMid slice"
              focusable="false"
              role="img"
              aria-label="Placeholder: 420x420"
            >
              <title>Placeholder</title>
              <rect width="100%" height="100%" fill="#eee" />
              <text x="50%" y="50%" fill="#aaa" dy=".3em">
                500x500
              </text>
            </svg>
          </div>
        </div>
        <hr class="featurette-divider" />
        <div>
          <h2 class="text-mutedtext text-center">
            <a
              href="https://github.com/leafinity/amazing_signt"
              style={{ color: 'orange' }}
            >
              <h2 class="featurette-heading text-center">Learn more!</h2>
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
