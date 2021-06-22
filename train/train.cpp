#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

using namespace std;
using namespace cv;


const int minimalnaPowierzchniaKonturu = 100;

const int nowaSzerokoscObrazu = 20;
const int nowaWysokoscObrazu = 30;


int main() {

    Mat obrazUczonychZnakow;         
    Mat obrazSkonwertowanyDoGrayscale;               
    Mat obrazWyblurowany;                 
    Mat obrazPoObrobce;                  
    Mat kopiaObrazuPoObrobce;              

    vector<vector<Point> > punktKonturu;       
    vector<Vec4i> Hierarcha;                    

    Mat wartosciPlikuClassificationsWFormacieInt;      
    Mat nauczoneObrazySkonwertowaneJakoFloat;

    obrazUczonychZnakow = imread("font.jpg");  

    
    vector<int> poprawneZnaki = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };
   

    if (obrazUczonychZnakow.empty()) {                               
        cout << "Nie mozna otworzyc obrazu z pliku." << endl;         
        return(0);                                                 
    }

    cvtColor(obrazUczonychZnakow, obrazSkonwertowanyDoGrayscale, COLOR_BGR2GRAY);        

    GaussianBlur(obrazSkonwertowanyDoGrayscale, obrazWyblurowany, Size(5, 5), 0);                                     

    adaptiveThreshold(obrazWyblurowany, obrazPoObrobce, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);                                     

    imshow("obrazPoObrobce", obrazPoObrobce);         

    kopiaObrazuPoObrobce = obrazPoObrobce.clone();          

    findContours(kopiaObrazuPoObrobce, punktKonturu, Hierarcha, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);               

    for (int i = 0; i < punktKonturu.size(); i++) {                           
        if (contourArea(punktKonturu[i]) > minimalnaPowierzchniaKonturu) {                
            Rect boundingRect = cv::boundingRect(punktKonturu[i]);                

            rectangle(obrazUczonychZnakow, boundingRect, Scalar(0, 0, 255), 2);      

            Mat matROI = obrazPoObrobce(boundingRect);           

            Mat nowyRozmiarROI;
            resize(matROI, nowyRozmiarROI, Size(nowaSzerokoscObrazu, nowaWysokoscObrazu));     

            imshow("matROI", matROI);                               
            imshow("nowyRozmiarROI", nowyRozmiarROI);                 
            imshow("obrazUczonychZnakow", obrazUczonychZnakow);       

            int intChar = waitKey(0);           

            if (intChar == 27) {        
                return(0);              
            }
            else if (find(poprawneZnaki.begin(), poprawneZnaki.end(), intChar) != poprawneZnaki.end()) {     

                wartosciPlikuClassificationsWFormacieInt.push_back(intChar);       

                Mat obrazPrzekonwertowanyNaFloat;                          
                nowyRozmiarROI.convertTo(obrazPrzekonwertowanyNaFloat, CV_32FC1);       

                Mat obrazZZaokraglonymiWartosciamiFloat = obrazPrzekonwertowanyNaFloat.reshape(1, 1);       

                nauczoneObrazySkonwertowaneJakoFloat.push_back(obrazZZaokraglonymiWartosciamiFloat);      
                                                                                            
            }   
        }   
    }   

    cout << "Uczenie zakonczone." << endl;

    FileStorage klasyfikacja("klasyfikacja.xml", FileStorage::WRITE);          

    if (klasyfikacja.isOpened() == false) {                                                        
        cout << "Nie mozna otworzyc pliku klasyfikacja.xml" <<endl;        
        return(0);                                                                                     
    }

    klasyfikacja << "klasyfikacja" << wartosciPlikuClassificationsWFormacieInt;        
    klasyfikacja.release();                                            

    FileStorage uczoneObrazy("obraz.xml", FileStorage::WRITE);         

    if (uczoneObrazy.isOpened() == false) {                                                 
        cout << "Nie mozna otworzyc pliku obraz.xml" << endl;        
        return(0);                                                                              
    }

    uczoneObrazy << "obraz" << nauczoneObrazySkonwertowaneJakoFloat;         
    uczoneObrazy.release();                                                 

    return(0);
}