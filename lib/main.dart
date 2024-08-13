import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'dart:io';
import 'dart:math';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show ByteData, rootBundle;
import 'package:path/path.dart' as path;

import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart'
    as helpsss;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(FaceRecognitionApp());
}

class FaceRecognitionApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: FaceRecognitionPage(),
    );
  }
}

class FaceRecognitionPage extends StatefulWidget {
  @override
  _FaceRecognitionPageState createState() => _FaceRecognitionPageState();
}

class _FaceRecognitionPageState extends State<FaceRecognitionPage> {
  File? _image;
  final picker = ImagePicker();
  final FaceDetector faceDetector = GoogleMlKit.vision.faceDetector();
  late tfl.Interpreter _interpreter;
  List<Face> faces = [];
  Map<int, List<double>> faceEmbeddings = {};
  Map<List<double>, String> knownEmbeddings =
      {}; // In-memory database of known faces
  List<File> croppedImages = [];
  List<Uint8List> tensorImages = [];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/facenet.tflite');
      print(
          "Model loaded successfully ${_interpreter.getInputTensor(0).shape}");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<void> _getImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    // final inputDetails = _interpreter.getInputTensor(0).shape;
    // print("Model input details: $inputDetails");

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await _detectFaces();
      await _recognizeFaces();
    }
  }

  Future<void> _detectFaces() async {
    final inputImage = InputImage.fromFile(_image!);
    final detectedFaces = await faceDetector.processImage(inputImage);

    setState(() {
      faces = detectedFaces;
    });

    if (detectedFaces.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("No faces detected.")),
      );
    }
  }

  Future<void> _recognizeFaces() async {
    for (var face in faces) {
      final cropRect = Rect.fromLTRB(
        face.boundingBox.left,
        face.boundingBox.top,
        face.boundingBox.right,
        face.boundingBox.bottom,
      );
      final croppedImage = await _cropImage(_image!, cropRect);
      final faceEmbedding = await _getFaceEmbedding(croppedImage);
      print("Face embedding: $faceEmbedding ${faceEmbeddings.toString()}");
      print(
          "slkadsf ${faceEmbeddings.toString()} ${knownEmbeddings.toString()}");
      print("known embeddings: ${knownEmbeddings.toString()}");

      setState(() {
        faceEmbeddings[face.trackingId ?? faces.indexOf(face)] = faceEmbedding;
        croppedImages.add(croppedImage);
      });

      final recognizedName = _findClosestEmbedding(faceEmbedding);

      if (recognizedName != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Recognized: $recognizedName")),
        );
      } else {
        // Prompt the user to tag the face
        await _showTaggingDialog(faceEmbedding);
      }
    }
  }

  Future<File> _cropImage(File imageFile, Rect cropRect) async {
    // Load the image file
    final image = img.decodeImage(await imageFile.readAsBytes());

    if (image == null) {
      throw Exception('Failed to decode image');
    }

    // Convert Rect to integers for cropping
    final int left = cropRect.left.toInt();
    final int top = cropRect.top.toInt();
    final int width = cropRect.width.toInt();
    final int height = cropRect.height.toInt();

    // Crop the image
    final croppedImage = img.copyCrop(image, left, top, width, height);

    // Save the cropped image to a new file
    final directory = path.dirname(imageFile.path);
    final newFilePath = path.join(directory,
        'cropped_${Random().nextInt(10)}_${path.basename(imageFile.path)}');
    final newFile = File(newFilePath);

    // Encode the image to PNG format
    final croppedImageBytes = img.encodePng(croppedImage);

    // Write the bytes to the new file
    await newFile.writeAsBytes(croppedImageBytes);

    print("Cropped image saved: $newFilePath");

    return newFile;
  }

  bool isTwoVectorsEqual(List<double> v1, List<double> v2) {
    for (int i = 0; i < v1.length; i++) {
      if (v1[i] != v2[i]) {
        return false;
      }
    }
    return true;
  }

  Uint8List imageToByteListFloat32(
      img.Image image, int inputSize, double mean, double std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (img.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (img.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Uint8List imageToByteListUint8(img.Image image, int inputSize) {
    var convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
    var buffer = Uint8List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = img.getRed(pixel);
        buffer[pixelIndex++] = img.getGreen(pixel);
        buffer[pixelIndex++] = img.getBlue(pixel);
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  Future<List<double>> _getFaceEmbedding(File croppedImage) async {
    print("Getting face embedding...");
    final inputImage = InputImage.fromFile(croppedImage);
    // Preprocess the image
    final tensorImage = _preprocess(inputImage);
    // img.Image image = img.decodeImage(croppedImage.readAsBytesSync())!;
    // img.Image image = img.decodeImage(_preprocess(inputImage))!;
    // final tensorImage = imageToByteListUint8(image, 160);

    setState(() {
      // tensorImages.add(tensorImage);
    });

    // final kkk = ResizeImage(imageProvider)
    print("Preprocessing done ${tensorImage.tensorBuffer.getShape()}");

    // Allocate tensors
    _interpreter.allocateTensors();

    // Prepare input buffer
    // ByteBuffer inputBuffer = tensorImage.buffer;
    // TensorBuffer inputBuffer = TensorBufferFloat([1, 160, 160, 3]);
    var inputShape = [1, 160, 160, 3]; // [batch_size, height, width, channels]
    var inputBuffer = Float32List.fromList(tensorImage.buffer.asFloat32List());
    print(
        "wtf ${inputBuffer.length} ${tensorImage.buffer.asFloat32List().length}");
    TensorBuffer inputTensorBuffer = TensorBufferFloat([1, 160, 160, 3]);
    // inputTensorBuffer.loadList(inputBuffer, shape: inputShape);

    // TensorBuffer inputBuffer = TensorBuffer.createFixedSize(
    //     [1, 160, 160, 3],
    //     TfLiteType.float32
    //   );
    // inputBuffer.loadBuffer(tensorImage.getTensorBuffer().buffer,
    //     shape: inputBuffer.getShape());

    // Allocate output buffer
    // var outputBuffer = List.filled(128, 0.0).reshape([1, 128]);
    // get output buffer in this shape [1, 512]
    // var outputBuffer = List.filled(512, 0.0).reshape([1, 512]);
    // Float32List outputBuffer = Float32List(512);
    // ByteBuffer outputBuffer = outputList.buffer;
    // TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
    //     [1, 512], tfl.TfLiteType.kTfLiteFloat32 as dynamic);

    // print(inputBuffer.asInt8List());
    // print(inputBuffer.toJS);

    print(
        "Input buffer: ${inputTensorBuffer.getShape()} okk1 aaa ${_interpreter.getInputTensor(0).shape} bbb ${_interpreter.getOutputTensor(0).shape}");
    var output = List.filled(1 * 512, 0).reshape([1, 512]);
    // var input = List.filled(1 * 160 * 160 * 3, 0).reshape([1, 160, 160, 3]);
    // // input[1] = tensorImage.getTensorBuffer().getShape();
    // int cc = 0;
    // for (int i = 0; i < 160; i++) {
    //   for (int j = 0; j < 160; j++) {
    //     for (int k = 0; k < 3; k++) {
    //       if (cc >= inputBuffer.length) {
    //         break;
    //       }
    //       input[0][i][j][k] = inputBuffer[cc];
    //       cc += 1;
    //     }
    //   }
    // }
    var input =
        tensorImage.getTensorBuffer().getDoubleList().reshape([1, 160, 160, 3]);
    // Run inference
    try {
      _interpreter.run(input, output);
    } catch (e) {
      print("Error running inference: $e");
    }

    print(
        "Output buffer: ${_interpreter.getInputTensor(0).shape} okk1 ${_interpreter.getOutputTensor(0).shape}");

    // Return the embedding vector
    // return outputBuffer.reshape([128]).cast<double>();
    // return outputBuffer.getDoubleList();
    print("Output bufferlklkj: ${output[0].length}");
    return output[0].cast<double>();
  }

  TensorImage _preprocess(InputImage inputImage) {
    final inputSize = 160; // Match this with your model's input size

    TensorImage tensorImage = TensorImage.fromFile(File(inputImage.filePath!));

    ImageProcessor imageProcessor = ImageProcessorBuilder()
        .add(ResizeOp(inputSize, inputSize, ResizeMethod.bilinear))
        .add(NormalizeOp(127.5, 127.5)) // Normalize to [-1, 1]
        .build();

    tensorImage = imageProcessor.process(tensorImage);
    return tensorImage;
    // input size: [1, 160, 160, 3]
  }

  String? _findClosestEmbedding(List<double> embedding) {
    double minDistance = 0.3;
    String? closestPerson;

    knownEmbeddings.forEach((knownEmbedding, personName) {
      print("fkkkkk ${isTwoVectorsEqual(knownEmbedding, embedding)}");
      double distance = _euclideanDistance(embedding, knownEmbedding);
      print("Distance: $distance");
      if (distance < minDistance) {
        minDistance = distance;
        closestPerson = personName;
      }
    });

    // Define a threshold for recognition
    if (minDistance < 1.0) {
      // Adjust this threshold as needed
      return closestPerson;
    } else {
      return null;
    }
  }

  double _euclideanDistance(List<double> v1, List<double> v2) {
    double sum = 0.0;
    for (int i = 0; i < v1.length; i++) {
      sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
  }

  Future<void> _showTaggingDialog(List<double> embedding) async {
    String? tagName;

    await showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("Tag this face"),
          content: TextField(
            onChanged: (value) {
              tagName = value;
            },
            decoration: InputDecoration(hintText: "Enter name"),
          ),
          actions: [
            TextButton(
              onPressed: () {
                if (tagName != null && tagName!.isNotEmpty) {
                  setState(() {
                    knownEmbeddings[embedding] = tagName!;
                  });
                }
                Navigator.of(context).pop();
              },
              child: Text("Tag"),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Face Recognition App"),
      ),
      body: ListView(
        children: [
          _image != null
              ? Image.file(_image!)
              : Container(height: 200, color: Colors.grey[300]),
          SizedBox(height: 20),
          ElevatedButton(
            onPressed: _getImage,
            child: Text("Select Image"),
          ),
          SizedBox(height: 20),
          faces.isNotEmpty
              ? Container(
                  height: 400,
                  child: ListView.builder(
                    itemCount: faces.length,
                    itemBuilder: (context, index) {
                      final face = faces[index];
                      final embedding =
                          faceEmbeddings[face.trackingId ?? index];
                      final name = embedding != null
                          ? _findClosestEmbedding(embedding)
                          : "Not recognized";
                      return ListTile(
                        leading: croppedImages.length > index
                            ? Image.file(croppedImages[index])
                            : null,
                        title: Text("Face ${index + 1}: $name"),
                      );
                    },
                  ),
                )
              : Container(),
        ],
      ),
    );
  }

  @override
  void dispose() {
    faceDetector.close();
    _interpreter.close();
    super.dispose();
  }
}
