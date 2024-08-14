import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;

import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
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
  Map<List<double>, String> knownEmbeddings =
      {}; // In-memory database of known faces
  List<File> croppedImages = [];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter =
          await tfl.Interpreter.fromAsset('assets/efficient_net.tflite');
      print(
          "Model loaded successfully ${_interpreter.getInputTensor(0).shape} output ${_interpreter.getOutputTensor(0).shape}");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<void> _getImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        croppedImages.clear();
        _image = File(pickedFile.path);
      });
      await _detectAndStoreCroppedFaces();
      // await _cropFaces();
      // await _recognizeFaces();
    }
  }

  Future<void> _detectAndStoreCroppedFaces() async {
    final inputImage = InputImage.fromFile(_image!);
    final detectedFaces = await faceDetector.processImage(inputImage);

    for (var face in detectedFaces) {
      final cropRect = Rect.fromLTRB(
        face.boundingBox.left,
        face.boundingBox.top,
        face.boundingBox.right,
        face.boundingBox.bottom,
      );
      final croppedImage = await _cropImage(_image!, cropRect);

      setState(() {
        croppedImages.add(croppedImage);
      });
    }

    if (detectedFaces.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No faces detected.")),
      );
    }
  }

  // Future<void> _cropFaces() async {
  //   for (var face in faces) {
  //     final cropRect = Rect.fromLTRB(
  //       face.boundingBox.left,
  //       face.boundingBox.top,
  //       face.boundingBox.right,
  //       face.boundingBox.bottom,
  //     );
  //     final croppedImage = await _cropImage(_image!, cropRect);
  //     // final faceEmbedding = await _getFaceEmbedding(croppedImage);

  //     setState(() {
  //       croppedImages.add(croppedImage);
  //     });

  //     // final recognizedName = _findClosestEmbedding(faceEmbedding);

  //     // if (recognizedName != null) {
  //     //   ScaffoldMessenger.of(context).showSnackBar(
  //     //     SnackBar(content: Text("Recognized: $recognizedName")),
  //     //   );
  //     // } else {
  //     //   // Prompt the user to tag the face
  //     //   await _showTaggingDialog(faceEmbedding);
  //     // }
  //   }
  // }

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
        'cropped_${Random().nextInt(10000)}_${path.basename(imageFile.path)}');
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
    final inputImage = InputImage.fromFile(croppedImage);
    final tensorImage = _preprocess(inputImage);
    _interpreter.allocateTensors();

    var output = List.filled(1 * 1000, 0).reshape([1, 1000]);
    var input =
        tensorImage.getTensorBuffer().getDoubleList().reshape([1, 224, 224, 3]);
    _interpreter.run(input, output);

    print("Output: ${output[0].length} ${output[0]}");
    return output[0].cast<double>();
  }

  Future<Map<String, dynamic>> _getFaceRecognitionData(
      File croppedImage) async {
    List<double> embedding = await _getFaceEmbedding(croppedImage);
    String closestPerson = _findClosestEmbedding(embedding) ?? "Unknown";
    print("problems ${embedding} $closestPerson");
    return {"embedding": embedding, "name": closestPerson};
  }

  TensorImage _preprocess(InputImage inputImage) {
    const inputSize = 224; // Match this with your model's input size

    TensorImage tensorImage = TensorImage.fromFile(File(inputImage.filePath!));

    ImageProcessor imageProcessor = ImageProcessorBuilder()
        .add(ResizeOp(inputSize, inputSize, ResizeMethod.bilinear))
        .add(NormalizeOp(127.5, 127.5)) // Normalize to [-1, 1]
        .build();

    tensorImage = imageProcessor.process(tensorImage);
    return tensorImage;
  }

  String? _findClosestEmbedding(List<double> embedding) {
    double minDistance = 0.019;
    String? closestPerson;

    knownEmbeddings.forEach((knownEmbedding, personName) {
      double distance = _euclideanDistance(embedding, knownEmbedding);
      if (distance < minDistance) {
        minDistance = distance;
        closestPerson = personName;
      }
    });

    print("Distance: $minDistance");
    if (minDistance < 0.019) {
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

  double _cosineSimilarity(List<double> vecA, List<double> vecB) {
    double dotProduct = 0.0;
    double magnitudeA = 0.0;
    double magnitudeB = 0.0;

    for (int i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magnitudeA += vecA[i] * vecA[i];
      magnitudeB += vecB[i] * vecB[i];
    }

    magnitudeA = sqrt(magnitudeA);
    magnitudeB = sqrt(magnitudeB);

    if (magnitudeA == 0 || magnitudeB == 0) return 0.0;

    final similarity = dotProduct / (magnitudeA * magnitudeB);
    return 1 - similarity;
  }

  Future<void> _showTaggingDialog(List<double> embedding) async {
    String? tagName;

    await showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text("Tag this face"),
          content: TextField(
            onChanged: (value) {
              tagName = value;
            },
            decoration: const InputDecoration(hintText: "Enter name"),
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
              child: const Text("Tag"),
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
        title: const Text("Face Recognition App"),
      ),
      body: ListView(
        children: [
          _image != null
              ? Image.file(_image!)
              : Container(height: 200, color: Colors.grey[300]),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _getImage,
            child: const Text("Select Image"),
          ),
          const SizedBox(height: 20),
          croppedImages.isNotEmpty
              ? Container(
                  height: 400,
                  child: ListView.builder(
                    itemCount: croppedImages.length,
                    itemBuilder: (context, index) {
                      // final face = faces[index];
                      // final embedding = faceEmbeddings[face.trackingId];
                      // final name = embedding != null
                      //     ? _findClosestEmbedding(embedding)
                      //     : "Not recognized";
                      File image = croppedImages[index];
                      // final embedding = _getFaceEmbedding(image);
                      // final name = "todo";
                      final details = _getFaceRecognitionData(image);
                      return FutureBuilder(
                          future: details,
                          key: UniqueKey(),
                          builder: (context, snapshot) {
                            if (snapshot.connectionState !=
                                ConnectionState.done) {
                              return ListTile(
                                leading: Image.file(image),
                                title: const Text("Recognizing..."),
                              );
                            }
                            return ListTile(
                              leading: Image.file(image),
                              title: Text(
                                  "Face ${index + 1}: ${snapshot.data!['name']}"),
                              onTap: () {
                                _showTaggingDialog(snapshot.data!['embedding']
                                    as List<double>);
                              },
                            );
                          });
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
