

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>

using namespace cv;
using namespace std;


int main() {

	VideoCapture capWebcam(0);

	char charCheckForEscKey = 0;
	auto font = FONT_HERSHEY_COMPLEX;

	if (capWebcam.isOpened() == false) {				
		std::cout << "error: capWebcam not accessed successfully\n\n";	
		return(0);														
	}

	Mat imgOriginal;
	Mat HSV;
	Mat imgThreshLow;
	Mat imgThreshHigh;
	Mat imgThresh;

	std::vector<std::vector<Point>> contours;
	std::vector<std::vector<Point>> contour0;

	double area = 0;
	size_t numLines = 0;

	while (charCheckForEscKey != 27 && capWebcam.isOpened()) {		
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);		

		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {		
			std::cout << "error: frame not read from webcam\n";		
			break;													
		}

		cvtColor(imgOriginal, HSV, COLOR_BGR2HSV);

		cv::inRange(HSV, cv::Scalar(0, 155, 155), cv::Scalar(18, 255, 255), imgThreshLow);
		cv::inRange(HSV, cv::Scalar(165, 155, 155), cv::Scalar(179, 255, 255), imgThreshHigh);

		cv::add(imgThreshLow, imgThreshHigh, imgThresh);

		cv::GaussianBlur(imgThresh, imgThresh, cv::Size(5, 5), 0);

		cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

		cv::erode(imgThresh, imgThresh, structuringElement);

		findContours(imgThresh,contour0 ,RETR_TREE, CHAIN_APPROX_SIMPLE);	
		
		contours.resize(contour0.size());

		for (size_t k = 0; k < contour0.size(); k++)
		{	
			approxPolyDP(Mat(contour0[k]),contours[k],arcLength(contour0[k],true)*0.01, true);
			
			area = contourArea(contours[k]);
			numLines = contours[k].size();
		}

		if (area > 3500) {
			drawContours(imgOriginal, contours, 0, Scalar(0, 255, 0), 2);
			if (numLines == 4) putText(imgOriginal, "Prostokat", Point(100,100), font, 1, Scalar(0, 0, 0));
			else if (numLines >= 3 && numLines <= 4) putText(imgOriginal, "Trojkat", Point(100, 100), font, 1, Scalar(0, 0, 0));
			else if (numLines > 10) putText(imgOriginal, "Kolo", Point(100, 100), font, 1, Scalar(0, 0, 0));
			
		}
		
		imshow("Obraz", imgOriginal);
		imshow("Mask", imgThresh);
		charCheckForEscKey = cv::waitKey(1);
	}

	return(0);
}

