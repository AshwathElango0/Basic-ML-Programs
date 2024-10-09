The file automatic_judge.py contains program for an application which uses a folder of images and compares each one to a reference image.

It displays the image with the most similarity in a Streamlit page, and forms a little leaderboard with the names of all the images and the similarity scores.

An EfficientNetB0 model is used to embed images and generate vectors for similarity checking.
