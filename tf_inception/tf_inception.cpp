/*
Sample of using OpenCV dnn module with Tensorflow Inception model.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const String keys =
"{help h    || Sample app for loading Inception TensorFlow model. "
"The model and class names list can be downloaded here: "
"https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip }"
"{model m   |tensorflow_inception_graph.pb| path to TensorFlow .pb model file }"
"{image i   || path to image file }"
"{i_blob    | input | input blob name) }"
"{o_blob    | softmax2 | output blob name) }"
"{c_names c | imagenet_comp_graph_label_strings.txt | path to file with classnames for class id }"
"{result r  || path to save output blob (optional, binary format, NCHW order) }"
;

/* 为blob找到最好的类 (最大概率类) */
void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

std::vector<String> readClassNames(const char *filename)
{
	std::vector<String> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();
	return classNames;
}

int main(int argc, char **argv)
{
	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String modelFile = "module_files/tensorflow_inception_graph.pb";//parser.get<String>("model");
	String imageFile = "test_image/badger.jpg";//parser.get<String>("image");
	String inBlobName = parser.get<String>("i_blob");
	String outBlobName = parser.get<String>("o_blob");

	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	String classNamesFile = "module_files/imagenet_comp_graph_label_strings.txt";//parser.get<String>("c_names");
	String resultFile = parser.get<String>("result");

	// Initialize network，从Tensorflow模型读取
	dnn::Net net = readNetFromTensorflow(modelFile);

	if (net.empty())
	{
		std::cerr << "Can't load network by using the mode file: " << std::endl;
		std::cerr << modelFile << std::endl;
		exit(-1);
	}

	// Prepare blob
	Mat img = imread(imageFile);
	if (img.empty())
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	// 转换Mat成批量图像
	Mat inputBlob = blobFromImage(img, 1.0f, Size(224, 224), Scalar(), true, false);
	inputBlob -= 117.0;

	// 设置网络输入
	net.setInput(inputBlob, inBlobName);

	// 计时器
	cv::TickMeter tm;
	tm.start();

	// 前向传播计算output
	Mat result = net.forward(outBlobName);

	tm.stop();

	if (!resultFile.empty()) {
		CV_Assert(result.isContinuous());

		ofstream fout(resultFile.c_str(), ios::out | ios::binary);
		fout.write((char*)result.data, result.total() * sizeof(float));
		fout.close();
	}

	std::cout << "Output blob shape " << result.size[0] << " x " << result.size[1] << " x " << result.size[2] << " x " << result.size[3] << std::endl;
	std::cout << "Inference time, ms: " << tm.getTimeMilli() << std::endl;
	
	if (!classNamesFile.empty()) {
		std::vector<String> classNames = readClassNames(classNamesFile.c_str());
		int classId;
		double classProb;
		//找到最相似的类
		getMaxClass(result, &classId, &classProb);
		//输出结果
		std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
		std::cout << "Probability: " << classProb * 100 << "%" << std::endl;

		const float* data = (float*)result.data;
		for (size_t i = 0; i < result.total(); i += 7)
		{
			//vector [id, classId, confidence, left, top, right, bottom]
			float confidence = data[i + 2];
			if (confidence > 0.8)
			{
				int left = max(0, min((int)data[i + 3], img.cols - 1));
				int top = max(0, min((int)data[i + 4], img.rows - 1));
				int right = max(0, min((int)data[i + 5], img.cols - 1));
				int bottom = max(0, min((int)data[i + 6], img.rows - 1));
				// 绘制标签，包含类名和概率
				String label = cv::format("%s, %.3f", classNames.at(classId).c_str(), confidence);
				int baseLine;
				Size labelSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				top = max(top, labelSize.height);
				putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			}
		}
	}

	imshow("frame", img);
	waitKey();
	return 0;
} //main
