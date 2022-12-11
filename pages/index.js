import { base64StringToBlob } from "blob-util";
import Head from "next/head";
import Image from "next/image";
import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";

export default function Home() {
  const [model, setModel] = useState({
    trained: false,
    loading: false,
    isError: false,
  });
  const [datasets, setDatasets] = useState([]);
  const imagesRef = useRef([]);

  const doTrainModel = async () => {
    const clf = doMakeModel();
    let datasetX = imagesRef.current.map((images, _) => {
      return images.map((image, _) => {
        return tf.browser.fromPixels(image);
      });
    });
    let datasetY = datasets.map((_, i) => {
      return Array.from({ length: datasetX[i].length }, (_, __) => i);
    });
    datasetX = datasetX.reduce((prev, next) => {
      return prev.concat(next);
    });
    datasetY = datasetY.reduce((prev, next) => {
      return prev.concat(next);
    });
    datasetY = tf.oneHot(tf.tensor1d(datasetY, "int32"), datasets.length);
    datasetX = tf.stack(datasetX);
    await clf.fit(datasetX, datasetY, {
      batchSize: 2,
      epochs: 10,
      shuffle: true,
      callbacks: {
        onEpochBegin: (e, _) => {
          console.log(e);
        },
      },
    });
  };

  const doMakeModel = () => {
    setModel({ ...model, loading: true });
    const clf = tf.sequential();
    clf.add(
      tf.layers.conv2d({
        inputShape: [96, 96, 3],
        dataFormat: "channelsLast",
        kernelSize: 3,
        filters: 3,
        strides: 1,
        activation: "relu",
      })
    );
    clf.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    clf.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 3,
        strides: 1,
        activation: "relu",
      })
    );
    clf.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    clf.add(tf.layers.dense({ units: 16, activation: "relu" }));
    clf.add(tf.layers.dense({ units: datasets.length, activation: "softmax" }));
    clf.add(tf.layers.flatten());
    clf.compile({
      optimizer: tf.train.sgd(0.01),
      loss: "categoricalCrossentropy",
      metrics: "accuracy",
    });
    return clf;
  };

  const doAddClass = () => {
    setDatasets([
      ...datasets,
      {
        name: "Untitled",
        datas: [],
      },
    ]);
    imagesRef.current = [...imagesRef.current, []];
  };

  const doDeleteClass = (e) => {
    imagesRef.current = imagesRef.current.filter(
      (_, i) => i !== parseInt(e.currentTarget.value)
    );
    setDatasets(
      datasets.filter((_, i) => i !== parseInt(e.currentTarget.value))
    );
  };

  const doChangeClassName = (e) => {
    const newDatasets = datasets.map((v, i) => {
      if (i === parseInt(e.currentTarget.name)) {
        v.name = e.currentTarget.value;
      }
      return v;
    });
    setDatasets(newDatasets);
  };

  const doAddDataToClass = ({ index, image }) => {
    const blobImage = base64StringToBlob(image.split(",")[1], "image/webp");
    const newDatasets = datasets.map((dataset, i) => {
      if (i !== index) {
        return dataset;
      }
      return {
        name: dataset.name,
        datas: [
          ...dataset.datas,
          {
            image: blobImage,
            url: URL.createObjectURL(blobImage),
            base64Image: image,
          },
        ],
      };
    });
    setDatasets(newDatasets);
    imagesRef.current[index] = [...imagesRef.current[index], null];
  };

  const doDeleteDataFromClass = (e) => {
    const indexParent = parseInt(e.currentTarget.name);
    const indexItem = parseInt(e.currentTarget.value);
    const newDatasets = datasets.map((dataset, i) => {
      if (i !== indexParent) {
        return dataset;
      }
      return {
        name: dataset.name,
        datas: dataset.datas.filter((_, i) => i !== indexItem),
      };
    });
    setDatasets(newDatasets);
    imagesRef.current[indexParent].filter((_, i) => i !== indexItem);
  };

  return (
    <div className="font-inter min-h-screen min-w-screen">
      <Head>
        <title>Teachable Machine by Amira and Afifudin</title>
      </Head>
      <div className="text-center p-2 border-b text-xs font-medium">
        Teachable Machine built by Amira and Afifudin
      </div>
      <div
        className="h-[calc(100vh-33px)] w-full grid"
        style={{ gridTemplateColumns: "1fr auto auto" }}
      >
        <div className="border-r">
          <div className="flex items-center justify-between gap-4 py-4 px-6">
            <p className="font-medium">Datasets</p>
            <button
              onClick={doAddClass}
              className="py-2 px-4 rounded bg-blue-50 hover:bg-blue-100 text-blue-700 transition-all text-sm font-medium"
            >
              Add Class
            </button>
          </div>
          {datasets.map((dataset, i) => {
            return (
              <div key={i} className="py-4 px-6 border-l border-y">
                <div className="flex items-center gap-4 mb-4">
                  <input
                    type="text"
                    name={i}
                    value={dataset.name}
                    onChange={doChangeClassName}
                    className="py-2 px-4 text-sm transition-all focus:shadow-xl outline-none rounded border w-full"
                  />
                  <button
                    value={i.toString()}
                    onClick={doDeleteClass}
                    className="py-2 px-4 rounded bg-red-100 hover:bg-red-200 text-red-700 transition-all font-medium text-sm"
                  >
                    Delete
                  </button>
                </div>
                <div
                  className="grid gap-8 overflow-x-auto"
                  style={{ gridTemplateColumns: "auto 1fr" }}
                >
                  <div>
                    <Webcam
                      imageSmoothing
                      videoConstraints={{ height: 224, width: 224 }}
                      className="rounded mb-2"
                    >
                      {({ getScreenshot }) => (
                        <button
                          value={i.toString()}
                          onClick={() => {
                            const imageSrc = getScreenshot();
                            doAddDataToClass({
                              index: parseInt(i),
                              image: imageSrc,
                            });
                          }}
                          className="py-2 px-4 w-full rounded bg-blue-100 hover:bg-blue-200 text-blue-700 transition-all font-medium text-sm"
                        >
                          Capture
                        </button>
                      )}
                    </Webcam>
                    <p className="text-xs font-medium mt-8 text-center">
                      {dataset.datas.length} items
                    </p>
                  </div>
                  <div className="grid grid-rows-3 grid-flow-col place-content-between place-self-start gap-3 h-full overflow-x-auto max-w-full p-2">
                    {dataset.datas.map((image, j) => {
                      return (
                        <div className="relative w-24 h-24 rounded" key={j}>
                          <button
                            name={i}
                            value={j}
                            onClick={doDeleteDataFromClass}
                            className="rounded-full bg-red-500 p-2 text-xs absolute -top-2 -right-2 z-10"
                          ></button>
                          <Image
                            src={image.url}
                            alt="image"
                            fill={true}
                            className="rounded"
                            ref={
                              imagesRef.current[i][j] !== undefined
                                ? (el) => (imagesRef.current[i][j] = el)
                                : null
                            }
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="grid" style={{ gridTemplateRows: "auto 1fr" }}>
          <div className="py-4 px-6 border-b">
            <p className="font-medium mb-4">Training</p>
            <button
              onClick={doTrainModel}
              className="py-2 px-4 rounded bg-blue-50 hover:bg-blue-100 text-blue-700 transition-all font-medium text-sm"
            >
              Train Model
            </button>
          </div>
          <div className="py-4 px-6 ">
            <p className="font-medium mb-4">Preview</p>
            <div className="grid grid-cols-2 rounded border-2 border-blue-700 mb-4 text-sm font-medium">
              <button className="p-2 m-1 rounded bg-blue-700 text-white">
                Camera
              </button>
              <button className="p-2 m-1 rounded bg-blue-50 text-blue-700">
                Input File
              </button>
            </div>
            {model.trained ? (
              <Webcam
                imageSmoothing
                videoConstraints={{ aspectRatio: 1 }}
                className="h-96 w-96 rounded"
              />
            ) : (
              <div className="h-96 w-96 rounded bg-neutral-800 text-white flex items-center justify-center">
                Train First
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
