// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList
{
  std::string image_name; //!< image file name
  int wait_t;             //!< waiting time
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

///////////////////////////////////////////////////////
// ZERO PADDING
///////////////////////////////////////////////////////

void addZeroPaddingGeneral(const cv::Mat &src, const cv::Mat &krnl, cv::Mat &padded, const cv::Point anchor, bool zeroPad = true)
{
  int padHTop = anchor.x;
  int padHBottom = krnl.rows - anchor.x - 1;
  int padWLeft = anchor.y;
  int padWRight = krnl.cols - anchor.y - 1;

  if (zeroPad)
  {
    padded = cv::Mat(src.rows + padHTop + padHBottom, src.cols + padWLeft + padWRight, CV_8UC1, cv::Scalar(0));
  }
  else // non crea problemi nel caso dell'erosione
  {
    padded = cv::Mat(src.rows + padHTop + padHBottom, src.cols + padWLeft + padWRight, CV_8UC1, cv::Scalar(255));
  }

  for (int v = padHTop; v < padded.rows - padHBottom; ++v)
  {
    for (int u = padWLeft; u < padded.cols - padWRight; ++u)
    {
      padded.at<u_char>(v, u) = src.at<u_char>(v - padHTop, u - padWLeft);
    }
  }

  return;
}

///////////////////////////////////////////////////////
// THRESHOLDING CON SOGLIA IDEALE
///////////////////////////////////////////////////////

void thresholding(const cv::Mat &image, cv::Mat &out)
{
  cv::Mat best_bin(image.rows, image.cols, CV_8UC1);

  int best_th;
  double best_dist = std::numeric_limits<double>::max();

  for (int t = 80; t <= 254; ++t)
  {
    cv::Mat bin(image.rows, image.cols, CV_8UC1);

    double abovem = 0;
    double belowm = 0;
    int abovec = 0;
    int belowc = 0;
    for (int j = 0; j < image.cols * image.rows; ++j)
    {
      if (image.data[j * image.channels()] >= t)
      {
        abovem += image.data[j * image.channels()];
        abovec++;
        bin.data[j] = 255;
      }
      else
      {
        if (image.data[j * image.channels()] >= 80)
        {
          belowm += image.data[j * image.channels()];
          belowc++;
          bin.data[j] = 0;
        }
      }
    }
    abovem /= abovec;
    belowm /= belowc;

    double sigmaa = 0;
    double sigmab = 0;
    for (int j = 0; j < image.cols * image.rows; ++j)
    {
      if (image.data[j * image.channels()] >= t)
      {
        sigmaa += (image.data[j * image.channels()] - abovem) * (image.data[j * image.channels()] - abovem);
      }
      else
      {
        if (image.data[j * image.channels()] >= 80)
          sigmab += (image.data[j * image.channels()] - belowm) * (image.data[j * image.channels()] - belowm);
      }
    }
    double dist = sigmaa * abovec / (abovec - 1) + sigmab * belowc / (belowc - 1);
    if (best_dist > dist)
    {
      best_dist = dist;
      best_th = t;
      bin.copyTo(best_bin);
    }
    // std::cout<<"th "<<t<<" dist "<<dist<<std::endl;
  }
  best_bin.copyTo(out);
  std::cout << "best th " << best_th << " best_dist " << best_dist << std::endl;
}

///////////////////////////////////////////////////////
// THRESHOLDING CON SOGLIA FISSATA
///////////////////////////////////////////////////////

void binarize(const cv::Mat &src, cv::Mat &out, int t)
{
  out = cv::Mat(src.rows, src.cols, CV_8UC1);
  for (int r = 0; r < out.rows; ++r)
  {
    for (int c = 0; c < out.cols; ++c)
    {
      if (src.at<u_char>(r, c) >= t)
      {
        out.at<u_char>(r, c) = 255;
      }
      else
      {
        out.at<u_char>(r, c) = 0;
      }
    }
  }
}

///////////////////////////////////////////////////////
// EROSIONE BINARIA
///////////////////////////////////////////////////////

void binaryErosion(const cv::Mat &src, cv::Mat &krnl, cv::Mat &out, const cv::Point anchor)
{
  cv::Mat padded;
  addZeroPaddingGeneral(src, krnl, padded, anchor);
  out = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
  bool diff;
  for (int r = 0; r < out.rows; ++r)
  {
    for (int c = 0; c < out.cols; ++c)
    {
      diff = false;
      for (int i = 0; i < krnl.rows; ++i)
      {
        for (int j = 0; j < krnl.cols; ++j)
        {
          if (krnl.data[j + i * krnl.cols] == 255)
          {
            if (krnl.data[j + i * krnl.cols] != padded.data[(c + i) + (r + j) * padded.cols])
            {
              diff = true;
              break;
            }
          }
        }

        if (diff)
          break;
      }
      if (!diff)
        out.at<u_char>(r, c) = 255;
    }
  }
}

///////////////////////////////////////////////////////
// ESPANSIONE BINARIA
///////////////////////////////////////////////////////

void binaryDilation(const cv::Mat &src, cv::Mat &krnl, cv::Mat &out, const cv::Point anchor)
{
  cv::Mat padded;
  addZeroPaddingGeneral(src, krnl, padded, anchor);
  out = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
  bool eq;

  for (int r = 0; r < out.rows; ++r)
  {
    for (int c = 0; c < out.cols; ++c)
    {
      eq = false;

      for (int i = 0; i < krnl.rows; ++i)
      {
        for (int j = 0; j < krnl.cols; ++j)
        {
          if (krnl.data[j + i * krnl.cols] == 255)
          {
            if (krnl.data[j + i * krnl.cols] == padded.data[(c + i) + (r + j) * padded.cols])
            {
              eq = true;
              break;
            }
          }
        }

        if (eq)
          break;
      }

      if (eq)
        out.at<u_char>(r, c) = 255;
    }
  }

  return;
}

///////////////////////////////////////////////////////
// CHIUSURA BINARIA
///////////////////////////////////////////////////////

void closingBinary(const cv::Mat &src, cv::Mat &krnl, cv::Mat &out, const cv::Point anchor)
{
  cv::Mat tmp;
  binaryDilation(src, krnl, tmp, anchor);
  binaryErosion(tmp, krnl, out, anchor);
}

///////////////////////////////////////////////////////
// APERTURA BINARIA
///////////////////////////////////////////////////////

void openingBinary(const cv::Mat &src, cv::Mat &krnl, cv::Mat &out, const cv::Point anchor)
{
  cv::Mat tmp;
  binaryErosion(src, krnl, tmp, anchor);
  binaryDilation(tmp, krnl, out, anchor);
}

///////////////////////////////////////////////////////
// COMPONENTI CONNESSE
///////////////////////////////////////////////////////

bool checkPresence(std::vector<int> &vector, int n)
{
  for (auto value : vector)
  {
    if (value == n)
    {
      return true;
    }
  }
  return false;
}

void insertEquivalences(std::vector<std::vector<int>> &equivalences, int n1, int n2)
{
  bool ok = false;
  for (size_t i = 0; i < equivalences.size(); i++)
  {
    if (checkPresence(equivalences[i], n1) && checkPresence(equivalences[i], n2))
    {
      ok = true;
      break;
    }
    else if (checkPresence(equivalences[i], n1) && !checkPresence(equivalences[i], n2))
    {
      ok = true;
      equivalences[i].push_back(n2);
      break;
    }
    else if (!checkPresence(equivalences[i], n1) && checkPresence(equivalences[i], n2))
    {
      ok = true;
      equivalences[i].push_back(n1);
      break;
    }
  }
  if (!ok)
  {
    std::vector<int> temp;
    temp.push_back(n1);
    temp.push_back(n2);
    equivalences.push_back(temp);
  }
}

bool checkEquivalences(std::vector<std::vector<int>> &equivalences, int n, int &min)
{
  for (size_t i = 0; i < equivalences.size(); i++)
  {
    if (checkPresence(equivalences[i], n))
    {
      min = *min_element(equivalences[i].begin(), equivalences[i].end());
      return true;
    }
  }
  return false;
}

void connectedComponents(const cv::Mat &src, cv::Mat &labels)
{
  labels = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
  // prima passata
  int label = 0; // contatore per assegnazione delle labels
  std::vector<std::vector<int>> equivalences;
  for (int r = 0; r < labels.rows; ++r)
  {
    for (int c = 0; c < labels.cols; ++c)
    {
      if (r == 0 && c == 0)
      { // sono nel primo pixel (controllo semplicemente se è uguale a 255 e in caso affermativo assegno una label)
        labels.at<u_char>(r, c) = label;
      }
      else if (r == 0 && c != 0)
      { // sono nella prima riga dal secondo pixel in poi (controllo solo i vicini di sinistra)
        if (src.at<u_char>(r, c) == src.at<u_char>(r, c - 1))
        {
          labels.at<u_char>(r, c) = labels.at<u_char>(r, c - 1);
        }
        else
        {
          label++;
          labels.at<u_char>(r, c) = label;
        }
      }
      else if (r != 0 && c == 0) // sono dalla seconda riga in poi ma nella prima colonna (controllo solo quello in alto)
      {

        if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c))
        {
          labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);
        }
        else
        {
          label++;
          labels.at<u_char>(r, c) = label;
        }
      }
      else
      { // sono negli altri casi

        if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) == src.at<u_char>(r, c - 1) && labels.at<u_char>(r - 1, c) == labels.at<u_char>(r, c - 1))
        { // stesso valore -> stessa label
          labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);
        }
        else if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) != src.at<u_char>(r, c - 1))
        {
          labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);
        }
        else if (src.at<u_char>(r, c) != src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) == src.at<u_char>(r, c - 1))
        {
          labels.at<u_char>(r, c) = labels.at<u_char>(r, c - 1);
        }
        else if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) == src.at<u_char>(r, c - 1) && labels.at<u_char>(r - 1, c) != labels.at<u_char>(r, c - 1))
        { // stesso valore -> label diversa
          labels.at<u_char>(r, c) = std::min(labels.at<u_char>(r - 1, c), labels.at<u_char>(r, c - 1));
          insertEquivalences(equivalences, labels.at<u_char>(r - 1, c), labels.at<u_char>(r, c - 1));
        }
        else
        {
          label++;
          labels.at<u_char>(r, c) = label;
        }
      }
    }
  }
  // seconda passata
  int min;
  for (int r = 0; r < labels.rows; ++r)
  {
    for (int c = 0; c < labels.cols; ++c)
    {
      if (checkEquivalences(equivalences, labels.at<u_char>(r, c), min))
      {
        labels.at<u_char>(r, c) = min;
      }
    }
  }
}

void connectedComponentsBinary(const cv::Mat &src, cv::Mat &labels)
{
  labels = cv::Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
  // prima passata
  int label = 0; // contatore per assegnazione delle labels
  std::vector<std::vector<int>> equivalences;
  for (int r = 0; r < labels.rows; ++r)
  {
    for (int c = 0; c < labels.cols; ++c)
    {
      if (r == 0 && c == 0)
      { // sono nel primo pixel (controllo semplicemente se è uguale a 255 e in caso affermativo assegno una label)
        if (src.at<u_char>(r, c) == 255)
        {
          labels.at<u_char>(r, c) = label;
        }
      }
      else if (r == 0 && c != 0)
      { // sono nella prima riga dal secondo pixel in poi (controllo solo i vicini di sinistra)
        if (src.at<u_char>(r, c) == 255)
        {
          if (src.at<u_char>(r, c) == src.at<u_char>(r, c - 1))
          {
            labels.at<u_char>(r, c) = labels.at<u_char>(r, c - 1);
          }
          else
          {
            label++;
            labels.at<u_char>(r, c) = label;
          }
        }
      }
      else if (r != 0 && c == 0) // sono dalla seconda riga in poi ma nella prima colonna (controllo solo quello in alto)
      {
        if (src.at<u_char>(r, c) == 255)
        {
          if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c))
          {
            labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);
          }
          else
          {
            label++;
            labels.at<u_char>(r, c) = label;
          }
        }
      }
      else
      { // sono negli altri casi
        if (src.at<u_char>(r, c) == 255)
        {
          if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) == src.at<u_char>(r, c - 1) && labels.at<u_char>(r - 1, c) == labels.at<u_char>(r, c - 1))
          { // stesso valore -> stessa label
            labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);
          }
          else if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) != src.at<u_char>(r, c - 1))
          {
            labels.at<u_char>(r, c) = labels.at<u_char>(r - 1, c);
          }
          else if (src.at<u_char>(r, c) != src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) == src.at<u_char>(r, c - 1))
          {
            labels.at<u_char>(r, c) = labels.at<u_char>(r, c - 1);
          }
          else if (src.at<u_char>(r, c) == src.at<u_char>(r - 1, c) && src.at<u_char>(r, c) == src.at<u_char>(r, c - 1) && labels.at<u_char>(r - 1, c) != labels.at<u_char>(r, c - 1))
          { // stesso valore -> label diversa
            labels.at<u_char>(r, c) = std::min(labels.at<u_char>(r - 1, c), labels.at<u_char>(r, c - 1));
            insertEquivalences(equivalences, labels.at<u_char>(r - 1, c), labels.at<u_char>(r, c - 1));
          }
          else
          {
            label++;
            labels.at<u_char>(r, c) = label;
          }
        }
      }
    }
  }
  // seconda passata
  int min;
  for (int r = 0; r < labels.rows; ++r)
  {
    for (int c = 0; c < labels.cols; ++c)
    {
      if (checkEquivalences(equivalences, labels.at<u_char>(r, c), min))
      {
        labels.at<u_char>(r, c) = min;
      }
    }
  }
}

void addPadding(const cv::Mat image, cv::Mat &out, int vPadding, int hPadding)
{
  out = cv::Mat(image.rows + vPadding * 2, image.cols + hPadding * 2, image.type(), cv::Scalar(0));

  for (int row = vPadding; row < out.rows - vPadding; ++row)
  {
    for (int col = hPadding; col < out.cols - hPadding; ++col)
    {
      for (int k = 0; k < out.channels(); ++k)
      {
        out.data[((row * out.cols + col) * out.elemSize() + k * out.elemSize1())] = image.data[(((row - vPadding) * image.cols + col - hPadding) * image.elemSize() + k * image.elemSize1())];
      }
    }
  }
}

void myfilter2D(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stridev, int strideh)
{

  if (!src.rows % 2 || !src.cols % 2)
  {
    std::cerr << "myfilter2D(): ERROR krn has not odd size!" << std::endl;
    exit(1);
  }

  int outsizey = (src.rows + (krn.rows / 2) * 2 - krn.rows) / (float)stridev + 1;
  int outsizex = (src.cols + (krn.cols / 2) * 2 - krn.cols) / (float)strideh + 1;
  out = cv::Mat(outsizey, outsizex, CV_32SC1);
  // std::cout << "Output image " << out.rows << "x" << out.cols << std::endl;

  cv::Mat image;
  addPadding(src, image, krn.rows / 2, krn.cols / 2);

  int xc = krn.cols / 2;
  int yc = krn.rows / 2;

  int *outbuffer = (int *)out.data;
  float *kernel = (float *)krn.data;

  for (int i = 0; i < out.rows; ++i)
  {
    for (int j = 0; j < out.cols; ++j)
    {
      int origy = i * stridev + yc;
      int origx = j * strideh + xc;
      float sum = 0;
      for (int ki = -yc; ki <= yc; ++ki)
      {
        for (int kj = -xc; kj <= xc; ++kj)
        {
          sum += image.data[(origy + ki) * image.cols + (origx + kj)] * kernel[(ki + yc) * krn.cols + (kj + xc)];
        }
      }
      outbuffer[i * out.cols + j] = sum;
    }
  }
}

void gaussianKrnl(float sigma, int r, cv::Mat &krnl)
{
  krnl = cv::Mat(2 * r + 1, 1, CV_32FC1, cv::Scalar(0.0));
  float sum = 0.0;
  for (int x = -r; x <= r; ++x)
  {
    krnl.at<float>(x + r, 0) = (exp(-pow(x, 2) / (2 * pow(sigma, 2)))) / (sqrt(CV_PI * 2) * sigma);
    sum += krnl.at<float>(x + r, 0);
  }
  krnl /= sum;
}

void gaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride = 1)
{
  cv::Mat gaussKrnl;
  gaussianKrnl(sigma, r, gaussKrnl);
  cv::Mat gaussKrnlT = gaussKrnl.t();

  // convolution
  cv::Mat tmp;
  // myfilter2D(src, gaussKrnl, tmp, 1, 1);
  cv::filter2D(src, tmp, CV_32F, gaussKrnl);
  cv::convertScaleAbs(tmp, tmp);
  cv::Mat outV;
  cv::filter2D(tmp, outV, CV_32F, gaussKrnlT);
  // myfilter2D(tmp, gaussKrnlT, outV, 1, 1);
  outV.convertTo(out, CV_8UC1);
  // cv::namedWindow("gaussian blur", cv::WINDOW_NORMAL);
  // cv::imshow("gaussian blur", out);
}

void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient)
{
  cv::Mat Ix, Iy;
  cv::Mat Kx = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  cv::Mat Ky = Kx.t();
  // myfilter2D(src, Kx, Ix, 1, 1);
  cv::filter2D(src, Ix, CV_32F, Kx);
  cv::filter2D(src, Iy, CV_32F, Ky);
  // myfilter2D(src, Ky, Iy, 1, 1);
  Ix.convertTo(Ix, CV_32FC1);
  Iy.convertTo(Iy, CV_32FC1);
  cv::pow(Ix.mul(Ix) + Iy.mul(Iy), 0.5, magn);
  orient = cv::Mat(Ix.size(), CV_32FC1);
  for (int r = 0; r < orient.rows; ++r)
  {
    for (int c = 0; c < orient.cols; ++c)
    {
      orient.at<float>(r, c) = atan2f(Iy.at<float>(r, c), Ix.at<float>(r, c));
    }
  }
  cv::Mat magnVis;
  cv::convertScaleAbs(magn, magnVis);
  cv::namedWindow("magnitude", cv::WINDOW_NORMAL);
  cv::imshow("magnitude", magnVis);
  cv::Mat adjMap;
  cv::convertScaleAbs(orient, adjMap, 255 / (2 * CV_PI));
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN); // COLORMAP_JET
  cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
  cv::imshow("sobel orientation", falseColorsMap);
}

template <class T>
float bilinear(const cv::Mat &src, float r, float c)
{
  // r in [0,rows-1] - c in [0,cols-1]
  if (r < 0 || r > (src.rows - 1) || c < 0 || c > (src.cols - 1))
    return -1;

  // get the largest possible integer less than or equal to r/c
  int rfloor = floor(r);
  int cfloor = floor(c);
  float t = r - rfloor;
  float s = c - cfloor;

  return (src.at<T>(rfloor, cfloor)) * (1 - s) * (1 - t) +
         (src.at<T>(rfloor, cfloor + 1)) * s * (1 - t) +
         (src.at<T>(rfloor + 1, cfloor)) * (1 - s) * t +
         (src.at<T>(rfloor + 1, cfloor + 1)) * t * s;
}

void findPeaks(const cv::Mat &magn, const cv::Mat &ori, cv::Mat &out)
{
  out = cv::Mat(magn.rows, magn.cols, CV_32FC1, cv::Scalar(0));

  // cv::Point e1, e2; //two neighboring points at distance 1 along the direction of the gradient
  float curr_val, theta, val_e1, val_e2;

  for (int r = 1; r < out.rows - 1; ++r)
  {
    for (int c = 1; c < out.cols - 1; ++c)
    {
      curr_val = magn.at<float>(r, c);
      theta = ori.at<float>(r, c);

      // e1 = cv::Point(c + 1 * cos(theta), r + 1 * sin(theta));
      // e2 = cv::Point(c - 1 * cos(theta), r - 1 * sin(theta));

      // val_e1 = bilinear(magn, e1.y, e1.x);
      // val_e2 = bilinear(magn, e2.y, e2.x);

      float e1x = c + 1 * cos(theta);
      float e1y = r + 1 * sin(theta);
      float e2x = c - 1 * cos(theta);
      float e2y = r - 1 * sin(theta);

      val_e1 = bilinear<float>(magn, e1y, e1x);
      val_e2 = bilinear<float>(magn, e2y, e2x);

      if (curr_val >= val_e1 && curr_val >= val_e2)
      {
        out.at<float>(r, c) = curr_val;
        // out.at<float>(e1.x, e1.y) = 0;
        // out.at<float>(e2.x, e2.y) = 0;
      }
      // else
      //   out.at<float>(r, c) = 0;
    }
  }
  cv::Mat vis;
  cv::convertScaleAbs(out, vis);
  cv::namedWindow("findpeaks", cv::WINDOW_NORMAL);
  cv::imshow("findpeaks", vis);
}

int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{

  cv::Mat first = cv::Mat(magn.size(), CV_8UC1);
  float p; // little optimization (complier should cope with this)
  if (t1 >= t2)
    return 1;

  int tm = t1 + (t2 - t1) / 2;

  std::vector<cv::Point2i> strong;
  std::vector<cv::Point2i> low;
  for (int r = 0; r < magn.rows; r++)
  {
    for (int c = 0; c < magn.cols; c++)
    {
      if ((p = magn.at<float>(r, c)) >= t2)
      {
        first.at<uint8_t>(r, c) = 255;
        strong.push_back(cv::Point2i(c, r)); // BEWARE at<>() and point2i() use a different coords order...
      }
      else if (p <= t1)
      {
        first.at<uint8_t>(r, c) = 0;
      }
      else
      {
        first.at<uint8_t>(r, c) = tm;
        low.push_back(cv::Point2i(c, r));
      }
    }
  }

  first.copyTo(out);

  // grow points > t2
  while (!strong.empty())
  {
    cv::Point2i p = strong.back();
    strong.pop_back();
    // std::cout << p.y << " " << p.x << std::endl;
    for (int ox = -1; ox <= 1; ++ox)
      for (int oy = -1; oy <= 1; ++oy)
      {
        int nx = p.x + ox;
        int ny = p.y + oy;
        if (nx > 0 && nx < out.cols && ny > 0 && ny < out.rows && out.at<uint8_t>(ny, nx) == tm)
        {
          // std::cerr << ".";
          out.at<uint8_t>(ny, nx) = 255;
          strong.push_back(cv::Point2i(nx, ny));
        }
      }
  }

  // wipe out residual pixels < t2
  while (!low.empty())
  {
    cv::Point2i p = low.back();
    low.pop_back();
    if (out.at<uint8_t>(p.y, p.x) < 255)
      out.at<uint8_t>(p.y, p.x) = 0;
  }

  cv::namedWindow("canny", cv::WINDOW_NORMAL);
  cv::imshow("canny", out);

  return 0;
}

///////////////////////////////////////////////////////
// HARRIS CORNER DETECTION
///////////////////////////////////////////////////////

void harrisCornerDetection(const cv::Mat &src, std::vector<cv::KeyPoint> &key_points, const float harris_th, const float alpha)
{
  cv::Mat vSobel = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  cv::Mat hSobel = (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
  cv::Mat Ix, Iy;
  cv::filter2D(src, Ix, CV_32F, vSobel);
  cv::filter2D(src, Iy, CV_32F, hSobel);

  // da usare nel caso di myfilter2D
  // Ix.convertTo(Ix, CV_32F);
  // Iy.convertTo(Iy, CV_32F);

  cv::Mat hGrad, vGrad;
  cv::convertScaleAbs(Ix, vGrad);
  cv::convertScaleAbs(Iy, hGrad);

  cv::Mat Ix_2 = cv::Mat(Ix.size(), Ix.type());
  cv::Mat Iy_2 = cv::Mat(Ix.size(), Ix.type());
  cv::Mat Ix_Iy = cv::Mat(Ix.size(), Ix.type());

  for (int r = 0; r < Ix.rows; ++r)
  {
    for (int c = 0; c < Ix.cols; ++c)
    {
      Ix_2.at<float>(r, c) = Ix.at<float>(r, c) * Ix.at<float>(r, c);
      Iy_2.at<float>(r, c) = Iy.at<float>(r, c) * Ix.at<float>(r, c);
      Ix_Iy.at<float>(r, c) = Ix.at<float>(r, c) * Iy.at<float>(r, c);
    }
  }

  Ix_2.convertTo(Ix_2, CV_8UC1);
  Iy_2.convertTo(Iy_2, CV_8UC1);
  Ix_Iy.convertTo(Ix_Iy, CV_8UC1);

  float sigma = 20;
  int kRadius = 1;

  cv::Mat g_Ix_2, g_Iy_2, g_Ix_Iy;
  gaussianBlur(Ix_2, sigma, kRadius, g_Ix_2);
  gaussianBlur(Iy_2, sigma, kRadius, g_Iy_2);
  gaussianBlur(Ix_Iy, sigma, kRadius, g_Ix_Iy);

  g_Ix_2.convertTo(g_Ix_2, CV_32F);
  g_Iy_2.convertTo(g_Iy_2, CV_32F);
  g_Ix_Iy.convertTo(g_Ix_Iy, CV_32F);

  cv::Mat thetas = cv::Mat(src.size(), g_Ix_2.type(), cv::Scalar(0));

  for (int r = 0; r < thetas.rows; ++r)
  {
    for (int c = 0; c < thetas.cols; ++c)
    {
      float g_Ix_2_value = g_Ix_2.at<float>(r, c);
      float g_Iy_2_value = g_Iy_2.at<float>(r, c);
      float g_Ix_Iy_value = g_Ix_Iy.at<float>(r, c);
      float det = (g_Ix_2_value * g_Iy_2_value) - pow(g_Ix_Iy_value, 2); // calcolo determinante
      float trace = (g_Ix_2_value + g_Iy_2_value);                       // calcolo traccia
      thetas.at<float>(r, c) = det - alpha * pow(trace, 2);
    }
  }

  // (simplified) non-maxima suppression

  int ngbSize = 3;
  double valMax;
  float currTheta;

  for (int r = ngbSize / 2; r < thetas.rows - ngbSize / 2; ++r)
  {
    for (int c = ngbSize / 2; c < thetas.cols - ngbSize / 2; ++c)
    {
      currTheta = thetas.at<float>(r, c);

      if (currTheta <= harris_th)
        thetas.at<float>(r, c) = 0;

      if (currTheta > harris_th)
      {
        cv::Mat ngb(thetas, cv::Rect(c - ngbSize / 2, r - ngbSize / 2, ngbSize, ngbSize));

        cv::minMaxIdx(ngb, NULL, &valMax, NULL, NULL);

        // for (int i = 0; i < ngb.rows; ++i)
        // {
        // 	for (int j = 0; j < ngb.cols; ++j)
        // 	{
        // 		if (ngb.at<float>(r, c) > valMax)
        // 		{
        // 			valMax = ngb.at<float>(r, c);
        // 		}
        // 	}
        // }

        if (currTheta < valMax)
          thetas.at<float>(r, c) = 0;
      }
    }
  }

  // display response matrix
  cv::Mat adjMap, falseColorsMap;
  double minR, maxR;

  cv::minMaxLoc(thetas, &minR, &maxR);
  cv::convertScaleAbs(thetas, adjMap, 255 / (maxR - minR));
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

  // save keypoints
  for (int r = 0; r < thetas.rows; ++r)
  {
    for (int c = 0; c < thetas.cols; ++c)
    {
      if (thetas.at<float>(r, c) > 0)
        key_points.push_back(cv::KeyPoint(c, r, 5));
    }
  }
}

int main(int argc, char **argv)
{
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;
  int imreadflags = cv::IMREAD_GRAYSCALE;

  std::cout << "Simple program." << std::endl;

  //////////////////////
  // parse argument list:
  //////////////////////
  ArgumentList args;
  if (!ParseInputs(args, argc, argv))
  {
    exit(0);
  }

  while (!exit_loop)
  {
    // generating file name
    //
    // multi frame case
    if (args.image_name.find('%') != std::string::npos)
      sprintf(frame_name, (const char *)(args.image_name.c_str()), frame_number);
    else // single frame case
      sprintf(frame_name, "%s", args.image_name.c_str());

    // opening file
    std::cout << "Opening " << frame_name << std::endl;

    cv::Mat image = cv::imread(frame_name, imreadflags);
    if (image.empty())
    {
      std::cout << "Unable to open " << frame_name << std::endl;
      return 1;
    }

    std::cout << "The image has " << image.channels() << " channels, the size is " << image.rows << "x" << image.cols << " pixels "
              << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl;

    //////////////////////
    // processing code here

    ///////////////////////////////////////////////////////
    /// MORFOLOGIA MATEMATICA
    ///////////////////////////////////////////////////////

    // cv::Mat thresholded;

    // // elemento strutturale

    // cv::Mat structuringElement(3, 3, CV_8UC1);
    // structuringElement.setTo(255);
    // cv::Point anchor(1, 1);

    // // thresholding(image, thresholded);
    // binarize(image, thresholded, 80);

    // cv::Mat eroded, dilated, closed, opened;

    // // erosione binaria

    // binaryErosion(thresholded, structuringElement, eroded, anchor);

    // // espansione binaria

    // binaryDilation(thresholded, structuringElement, dilated, anchor);

    // // chiusura binaria

    // closingBinary(thresholded, structuringElement, closed, anchor);

    // // apertura binaria

    // openingBinary(thresholded, structuringElement, opened, anchor);

    // // componenti connesse su opened binary

    // cv::Mat prova = (cv::Mat_<u_char>(10, 10) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //                  0, 255, 255, 0, 0, 255, 255, 255, 255, 0,
    //                  0, 255, 255, 0, 255, 255, 255, 255, 255, 0,
    //                  0, 255, 255, 255, 255, 0, 0, 0, 255, 0,
    //                  0, 255, 255, 0, 0, 255, 0, 0, 255, 0,
    //                  0, 255, 255, 0, 0, 0, 0, 0, 255, 0,
    //                  0, 255, 255, 0, 0, 0, 0, 0, 255, 0,
    //                  0, 255, 255, 255, 255, 255, 255, 255, 255, 0,
    //                  0, 255, 255, 255, 255, 255, 255, 255, 255, 0,
    //                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    // cv::Mat labels;

    // std::cout << prova << std::endl;

    // connectedComponentsBinary(prova, labels);

    // std::cout << labels << std::endl;

    /////////////////////

    // cv::Mat magnitude, orientation, blurred, thinned, thresholded;

    // gaussianBlur(image, 1, 1, blurred);
    // sobel3x3(blurred, magnitude, orientation);
    // findPeaks(magnitude, orientation, thinned);
    // doubleTh(thinned, thresholded, 90, 150);

    std::vector<cv::KeyPoint> key_points;
    float alpha = 0.04f;

    int th = 1;

    harrisCornerDetection(image, key_points, th, alpha);

    std::cout << key_points.size() << std::endl;

    // display image
    cv::namedWindow("original image", cv::WINDOW_NORMAL);
    cv::imshow("original image", image);

    cv::Mat k_points;
    cv::drawKeypoints(image, key_points, k_points, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // display image
    cv::namedWindow("k_points", cv::WINDOW_NORMAL);
    cv::imshow("k_points", k_points);

    // cv::namedWindow("thresholded image", cv::WINDOW_NORMAL);
    // cv::imshow("thresholded image", thresholded);

    // cv::namedWindow("eroded binary image", cv::WINDOW_NORMAL);
    // cv::imshow("eroded binary image", eroded);

    // cv::namedWindow("dilated binary image", cv::WINDOW_NORMAL);
    // cv::imshow("dilated binary image", dilated);

    // cv::namedWindow("closed binary image", cv::WINDOW_NORMAL);
    // cv::imshow("closed binary image", closed);

    // cv::namedWindow("opened binary image", cv::WINDOW_NORMAL);
    // cv::imshow("opened binary image", opened);

    // wait for key or timeout
    unsigned char key = cv::waitKey(args.wait_t);
    std::cout << "key " << int(key) << std::endl;

    // here you can implement some looping logic using key value:
    //  - pause
    //  - stop
    //  - step back
    //  - step forward
    //  - loop on the same frame

    switch (key)
    {
    case 'p':
      std::cout << "Mat = " << std::endl
                << image << std::endl;
      break;
    case 'q':
      exit_loop = 1;
      break;
    case 'c':
      std::cout << "SET COLOR imread()" << std::endl;
      imreadflags = cv::IMREAD_COLOR;
      break;
    case 'g':
      std::cout << "SET GREY  imread()" << std::endl;
      imreadflags = cv::IMREAD_GRAYSCALE; // Y = 0.299 R + 0.587 G + 0.114 B
      break;
    }

    frame_number++;
  }

  return 0;
}

#if 0
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|in.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0     |wait before next frame (ms)}"
      "{help    h|<none>|produce help message}"
      );

  if(parser.has("help"))
  {
    parser.printMessage();
    return false;
  }

  args.image_name = parser.get<std::string>("input");
  args.wait_t     = parser.get<int>("wait");

  return true;
}
#else

#include <unistd.h>
bool ParseInputs(ArgumentList &args, int argc, char **argv)
{
  int c;

  while ((c = getopt(argc, argv, "hi:t:")) != -1)
    switch (c)
    {
    case 't':
      args.wait_t = atoi(optarg);
      break;
    case 'i':
      args.image_name = optarg;
      break;
    case 'h':
    default:
      std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
      std::cout << "exit:  type q" << std::endl
                << std::endl;
      std::cout << "Allowed options:" << std::endl
                << "   -h                       produce help message" << std::endl
                << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
                << "   -t arg                   wait before next frame (ms)" << std::endl
                << std::endl;
      return false;
    }
  return true;
}

#endif
