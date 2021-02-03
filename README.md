# ADR
Action-conditioned disentangled representations for video prediction


(Code is being refactored from previous version)

--------------------

[[Paper]](https://drive.google.com/file/d/1idy9Bm53dW1zCUJ7epuugUwrPzd-oulV/view?usp=sharing) | [[Thesis]](https://drive.google.com/file/d/1QpDfYpAEwYPErHz6YYG3uNJsH-Nj9pVa/view?usp=sharing)

### Overview
In this work we take on the challenge of overcoming one of the main hurdles in the field of video prediction which is the prediction of object movement. 
The key insight of our solution is that, in a robotic scenario, it should be easy to predict the agent's own movement, and that the supression of that prediction should allow the model to focus on the more difficult to predict movement of the objects.

With this in mind, we first propose **ADR-AO** (agent only), which predicts the future pose of the agent from the observation of a few context frames and the knowledge of the agent's future actions, while explicitly ignoring the objects.

<img src="https://web.ist.utl.pt/ist181063/vp_examples/adr_ao/ex1/gt.gif" width="150" height="150"/> <img src="https://web.ist.utl.pt/ist181063/vp_examples/adr_ao/ex1/x_a.gif" width="150" height="150"/>

The error between ground truth frames and the frames generated with ADR-AO produces an image dominated by information of the objects that move during the video. This is a cue on object information obtained in a self-supervised way, without the need for data pre-processing or human annotation.

<img src="https://web.ist.utl.pt/ist181063/vp_examples/error_images/example_a/gt.gif" width="150" height="150"/> <img src="https://web.ist.utl.pt/ist181063/vp_examples/error_images/example_a/x_a.gif" width="150" height="150"/> <img src="https://web.ist.utl.pt/ist181063/vp_examples/error_images/example_a/err.gif" width="150" height="150"/>

The error images can be used to learn a representation of the objects. When predicting into the future with **ADR-VP**, an LSTM receives a content, action and object representation at time-step *t* and outputs the object representation for *t+1*. Predictions can then be fed back into the network, allowing the model to hallucinate the future.
