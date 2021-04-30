#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

using namespace std;
using namespace cv;

const int minimalnaPowierzchniaKonturu = 100;

const int nowaSzerokoscObrazu = 20;
const int nowaWysokoscObrazu = 30;


class KonturZDanymi {
public:
    
    vector<Point> punktKonturu;          
    Rect boundingRect;                      
    float powierzchnia;                              

    bool sprawdzCzyKonturJestPoprawny() {                              
        if (powierzchnia < minimalnaPowierzchniaKonturu) return false;           
        return true;                                            
    }

    
    static bool posortujBoundingRectPoPozycjiX(const KonturZDanymi& cwdLeft, const KonturZDanymi& cwdRight) {      
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   
    }

};


int main() {


    vector<KonturZDanymi> wszystkieKonturyZDanymi;           
    vector<KonturZDanymi> poprawneKonturyZDanymi;  

    vector<vector<Point> > punktKonturu;        
    vector<Vec4i> Hierarcha;

    Mat obrazSkonwertowanyDoGrayscale;         
    Mat obrazWyblurowany;             
    Mat obrazPoObrobce;              
    Mat kopiaObrazuPoObrobce;  

    Mat nauczoneObrazySkonwertowaneJakoFloat; 

    Mat wartosciPlikuClassificationsWFormacieInt;  

    string numeryTablicy; 

    Mat obrazTablicyRejestracyjnej =  imread("test.jpg");   

    FileStorage klasyfikacja("klasyfikacja.xml", FileStorage::READ);        
    if (klasyfikacja.isOpened() == false) {                                                    
        cout << "Blad przy otwarciu pliku klasyfikacja.xml" << endl;    
        return(0);                                                                                 
    }

    klasyfikacja["classifications"] >> wartosciPlikuClassificationsWFormacieInt;      
    klasyfikacja.release();                                        

    FileStorage uczoneObrazy("obraz.xml", FileStorage::READ);         

    if (uczoneObrazy.isOpened() == false) {                                                 
        cout << "Blad przy otwarciu pliku obraz.xml" << endl;        
        return(0);                                                                              
    }

    uczoneObrazy["images"] >> nauczoneObrazySkonwertowaneJakoFloat;           
    uczoneObrazy.release();                                                 

    Ptr<ml::KNearest>  kNearest(ml::KNearest::create());            
                                                                                
    kNearest->train(nauczoneObrazySkonwertowaneJakoFloat, ml::ROW_SAMPLE, wartosciPlikuClassificationsWFormacieInt);

    if (obrazTablicyRejestracyjnej.empty()) {                                
        cout << "Nie mozna otowrzyc obrazu."<< endl;         
        return(0);                                                  
    }
         

    cvtColor(obrazTablicyRejestracyjnej, obrazSkonwertowanyDoGrayscale, COLOR_BGR2GRAY);         
  
    GaussianBlur(obrazSkonwertowanyDoGrayscale, obrazWyblurowany, Size(5, 5), 0);                        
                              
    adaptiveThreshold(obrazWyblurowany, obrazPoObrobce, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);                                   

    kopiaObrazuPoObrobce = obrazPoObrobce.clone();                         

    findContours(kopiaObrazuPoObrobce, punktKonturu, Hierarcha, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);               

    for (int i = 0; i < punktKonturu.size(); i++) {               
        KonturZDanymi KonturZDanymi;                                                    
        KonturZDanymi.punktKonturu = punktKonturu[i];                                          
        KonturZDanymi.boundingRect = boundingRect(KonturZDanymi.punktKonturu);         
        KonturZDanymi.powierzchnia = contourArea(KonturZDanymi.punktKonturu);               
        wszystkieKonturyZDanymi.push_back(KonturZDanymi);                                     
    }

    for (int i = 0; i < wszystkieKonturyZDanymi.size(); i++) {                      
        if (wszystkieKonturyZDanymi[i].sprawdzCzyKonturJestPoprawny()) {                   
            poprawneKonturyZDanymi.push_back(wszystkieKonturyZDanymi[i]);            
        }
    }
    
    sort(poprawneKonturyZDanymi.begin(), poprawneKonturyZDanymi.end(), KonturZDanymi::posortujBoundingRectPoPozycjiX);
        

    for (int i = 0; i < poprawneKonturyZDanymi.size(); i++) {            

                                                                        
        rectangle(obrazTablicyRejestracyjnej, poprawneKonturyZDanymi[i].boundingRect, Scalar(0, 255, 0), 2);                                           

        Mat matROI = obrazPoObrobce(poprawneKonturyZDanymi[i].boundingRect);          

        Mat nowyRozmiarROI;
        resize(matROI, nowyRozmiarROI, Size(nowaSzerokoscObrazu, nowaWysokoscObrazu));     

        Mat ROIWFormacieFloat;
        nowyRozmiarROI.convertTo(ROIWFormacieFloat, CV_32FC1);             

        Mat skonwertowanyROIDoFormatuFloat = ROIWFormacieFloat.reshape(1, 1);

        Mat obecnyZnak(0, 0, CV_32F);

        kNearest->findNearest(skonwertowanyROIDoFormatuFloat, 1, obecnyZnak);     

        float obecnyZnakWFormacieFloat = (float)obecnyZnak.at<float>(0, 0);

        numeryTablicy = numeryTablicy + char(int(obecnyZnakWFormacieFloat));        
    }

    numeryTablicy.erase(numeryTablicy.begin());                           
    cout  << "Tablica rejestracyjna o numerach:  " << numeryTablicy << endl;       

    imshow("obrazTablicyRejestracyjnej", obrazTablicyRejestracyjnej);     

    waitKey(0);                                        

    return(0);
}