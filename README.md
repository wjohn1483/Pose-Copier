# Pose-Copier

This repository stores the source codes of pose copier which was modified from the example of [post-detection](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection).

By using the commands below, you can generate a page that is able to evalute how close the pose between the image stream captured from camera and the given photo.

## To Build Dependencies

The command below will install `yarn` and the dependencies that are required for building the html.

```bash
make dependency
```

## To Generate HTML

This will use `parcel` to compile html and serve it at `localhost:1234`.

```bash
make run
```

The html outputs will be stored in folder **dist/**.

## To Open HTML

We need to put which model we would like to use in the url and this command opens the url contains model name for you.

```bash
make open
```
