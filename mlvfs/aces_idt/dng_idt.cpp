#include "dng_idt.h"

#include <cmath>
#include <algorithm>
#include <functional>
#include <Eigen/Core>
#include <ceres/ceres.h>

/*
 * Code borrowed from rawtoaces
 * https://github.com/ampas/rawtoaces
 */

using namespace std;

// Roberson UV Table
static const double Robertson_uvtTable[][3] = {
    { 0.18006,  0.26352,  -0.24341 },
    { 0.18066,  0.26589,  -0.25479 },
    { 0.18133,  0.26846,  -0.26876 },
    { 0.18208,  0.27119,  -0.28539 },
    { 0.18293,  0.27407,  -0.3047 },
    { 0.18388,  0.27709,  -0.32675 },
    { 0.18494,  0.28021,  -0.35156 },
    { 0.18611,  0.28342,  -0.37915 },
    { 0.18740,  0.28668,  -0.40955 },
    { 0.18880,  0.28997,  -0.44278 },
    { 0.19032,  0.29326,  -0.47888 },
    { 0.19462,  0.30141,  -0.58204 },
    { 0.19962,  0.30921,  -0.70471 },
    { 0.20525,  0.31647,  -0.84901 },
    { 0.21142,  0.32312,  -1.0182 },
    { 0.21807,  0.32909,  -1.2168 },
    { 0.22511,  0.33439,  -1.4512 },
    { 0.23247,  0.33904,  -1.7298 },
    { 0.24010,  0.34308,  -2.0637 },
    { 0.24792,  0.34655,  -2.4681 },
    { 0.25591,  0.34951,  -2.9641 },
    { 0.26400,  0.35200,  -3.5814 },
    { 0.27218,  0.35407,  -4.3633 },
    { 0.28039,  0.35577,  -5.3762 },
    { 0.28863,  0.35714,  -6.7262 },
    { 0.29685,  0.35823,  -8.5955 },
    { 0.30505,  0.35907,  -11.324 },
    { 0.31320,  0.35968,  -15.628 },
    { 0.32129,  0.36011,  -23.325 },
    { 0.32931,  0.36038,  -40.77  },
    { 0.33724,  0.36051,  -116.45 }
};


// Roberson Mired Matrix
static const double RobertsonMired[] = {
    1.0e-10, 10.0, 20.0, 30.0, 40.0,
    50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    125.0, 150.0, 175.0, 200.0, 225.0,
    250.0, 275.0, 300.0, 325.0, 350.0,
    375.0, 400.0, 425.0, 450.0, 475.0,
    500.0, 525.0, 550.0, 575.0, 600.0f
};

static const double XYZ_acesrgb_3[3][3] = {
    { 1.0634731317028,      0.00639793641966071,   -0.0157891874506841 },
    { -0.492082784686793,   1.36823709310019,      0.0913444629573544  },
    { -0.0028137154424595,  0.00463991165243123,   0.91649468506889    }
};

static const double AP0toAP1[3][3] = {
	{1.4514393161, -0.2365107469, -0.2149285693},
    {-0.0765537734, 1.1762296998, -0.0996759264},
	{0.0083161484, -0.0060324498, 0.9977163014}
};

static const double chromaticitiesACES[4][2] = {
    { 0.73470,  0.26530 },
    { 0.00000,  1.00000 },
    { 0.00010,  -0.07700},
    { 0.32168,  0.33767 }
};

//  Color Adaptation Matrices - Cat02 (default)
static const double cat02[3][3] = {
    {0.7328,  0.4296,  -0.1624},
    {-0.7036, 1.6975,  0.0061 },
    {0.0030,  0.0136,  0.9834 }
};

#define sign(x)		((x) > 0 ? 1 : ( (x) < 0 ? (0-1) : 0))
#define FORI(val) for (int i=0; i < val; i++)
#define FORJ(val) for (int j=0; j < val; j++)
#define FORIJ(val1, val2) for (int i=0; i < val1; i++) for (int j=0; j < val2; j++)
#define countSize(a)    (  static_cast<int> (sizeof(a) / sizeof((a)[0])) )

template <typename T>
vector < T > addVectors ( const vector < T > & vectorA, const vector < T > & vectorB ) {
    assert ( vectorA.size() == vectorB.size() );
    vector < T > sum;
    sum.reserve ( vectorA.size() );
    std::transform ( vectorA.begin(), vectorA.end(),
                     vectorB.begin(), std::back_inserter (sum),
                     std::plus<T>() );
    return sum;
};

// This is not the typical "cross" product
template <typename T>
T cross2 ( const vector <T> &vectorA, const vector < T > & vectorB ) {
    assert (vectorA.size() == 2 && vectorB.size() == 2 );
    return vectorA[0] * vectorB[1] - vectorA[1] * vectorB[0];
};

template<typename T>
int isSquare ( const vector < vector < T > > & vm ) {
    FORI(vm.size()){
        if (vm[i].size() != vm.size())
            return 0;
    }

    return 1;
};

template <typename T>
vector < vector <T> > invertVM ( const vector < vector < T > > & vMtx ) {
    assert(isSquare(vMtx));

    Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic > m;
    m.resize(vMtx.size(), vMtx[0].size());
    FORIJ(m.rows(), m.cols()) m(i,j) = vMtx[i][j];

    //    Map < Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic, RowMajor > > m (vMtx[0]);
    //    m.resize(vMtx.size(), vMtx[0].size());

    m = m.inverse();

    vector < vector <T> > vMtxR (m.rows(), vector<T>(m.cols()));
    FORIJ(m.rows(), m.cols()) vMtxR[i][j] = m(i, j);

    return vMtxR;
};


template <typename T>
vector < T > invertV ( const vector < T > & vMtx ) {
    int size = std::sqrt ( static_cast<int> (vMtx.size()) );
    vector < vector <T> > tmp ( size, vector <T> (size) );

    FORIJ ( size, size )
        tmp[i][j] = vMtx[i*size+j];

    tmp = invertVM ( tmp );
    vector <T> result (vMtx.size());

    FORIJ ( size, size ) result[i*size+j] = tmp[i][j];

    return result;
};

template <typename T>
void scaleVector ( vector < T > & vct, const T scale ) {
    Eigen::Matrix <T, Eigen::Dynamic, 1> v;
    v.resize(vct.size(), 1);

    FORI(vct.size()) v(i,0) = vct[i];
    v *= scale;

    FORI(vct.size()) vct[i] = v(i,0);

    return;
};

template <typename T>
T sumVectorM ( const vector < vector < T > > & vct ) {
    int row = vct.size();
    int col = vct[0].size();

    T sum = T(0);
    Eigen::Matrix <T, Eigen::Dynamic, 1> v;
    v.resize ( row * col, 1 );

    FORIJ ( row, col )
        v(i*col + j) = vct[i][j];

    return v.sum();
};

template <typename T>
vector < T > subVectors ( const vector <T> &vectorA, const vector < T > & vectorB ) {
    assert ( vectorA.size() == vectorB.size() );
    vector < T > diff;
    diff.reserve ( vectorA.size() );
    std::transform ( vectorA.begin(), vectorA.end(),
                     vectorB.begin(), std::back_inserter (diff),
                     std::minus<T>() );
    return diff;
};

template <typename T>
vector<T> mulVectorElement ( const vector < T > & vct1,
                             const vector < T > & vct2 ) {
    assert(vct1.size() == vct2.size());

    Eigen::Array <T, Eigen::Dynamic, 1> a1, a2;
    a1.resize(vct1.size(), 1);
    a2.resize(vct1.size(), 1);

    FORI ( a1.rows() ) {
        a1(i, 0) = vct1[i];
        a2(i, 0) = vct2[i];
    }
    a1 *= a2;

    vector < T > vct3 ( a1.data(),
                        a1.data() + a1.rows() * a1.cols() );

    return vct3;
};

template <typename T>
vector<T> divVectorElement ( const vector < T > & vct1,
                             const vector < T > & vct2 ) {
    assert(vct1.size() == vct2.size());

    vector<T> vct2D (vct2.size(), T(1.0));
    FORI(vct2.size()) {
        assert(vct2[i] != T(0.0));
        vct2D[i] = T(1.0) / vct2[i];
    }

    return mulVectorElement ( vct1, vct2D );
};

template <typename T>
vector < T > mulVector ( vector < T > vct1, vector < T > vct2, int k = 3 )
{
    int rows = ( static_cast < int > ( vct1.size() ) ) / k;
    int cols = ( static_cast < int > ( vct2.size() ) ) / k;

    assert ( rows * k == vct1.size() );
    assert ( k * cols == vct2.size() );

    vector < T > vct3 (rows * cols);
    T * pA = &vct1[0];
    T * pB = &vct2[0];
    T * pC = &vct3[0];

    for ( int r = 0; r < rows; r++ ) {
        for ( int cArB = 0; cArB < k; cArB++ ) {
            for ( int c = 0; c < cols; c++ ) {
                pC[r * cols + c] += pA[r * k + cArB] * pB[cArB * cols + c];
            }
        }
    }

    return vct3;
};

template <typename T>
vector < vector < T > > mulVector ( const vector < vector < T > > & vct1,
                                    const vector < vector < T > > & vct2 ) {
    assert(vct1.size() != 0 && vct2.size() != 0);

    Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic> m1, m2, m3;
    m1.resize(vct1.size(), vct1[0].size());
    m2.resize(vct2[0].size(), vct2.size());

    FORIJ (m1.rows(), m1.cols())
        m1(i, j) = vct1[i][j];
    FORIJ (m2.rows(), m2.cols())
        m2(i, j) = vct2[j][i];

    m3 = m1 * m2;

    vector < vector<T> > vct3( m3.rows(), vector < T > ( m3.cols() ) );
    FORIJ (m3.rows(), m3.cols()) vct3[i][j] = m3(i, j);

    return vct3;
};

template <typename T>
vector < T > mulVector ( const vector < vector < T > > & vct1,
                         const vector < T > & vct2 ) {
    assert ( vct1.size() != 0 &&
             (vct1[0]).size() == vct2.size() );

    Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic> m1, m2, m3;
    m1.resize(vct1.size(), vct1[0].size());
    m2.resize(vct2.size(), 1);

    FORIJ (m1.rows(), m1.cols())
        m1(i, j) = vct1[i][j];
    FORI (m2.rows())
        m2(i, 0) = vct2[i];

    m3 = m1 * m2;

    vector< T > vct3 (m3.data(),
                      m3.data() + m3.rows() * m3.cols());

    return vct3;
};

template <typename T>
vector < T > mulVector ( const vector < T > & vct1,
                         const vector < vector < T > > & vct2 ) {
    return mulVector (vct2, vct1);
};

template <typename T>
T sumVector ( const vector < T > & vct ) {
    Eigen::Matrix <T, Eigen::Dynamic, 1> v;
    v.resize(vct.size(), 1);
    FORI(v.rows()) v(i, 0) = vct[i];

    return v.sum();
};

template <typename T>
vector < T > diagV ( const vector < T > & vct ) {
    assert( vct.size() != 0 );

    int length = static_cast<int>( vct.size() );
    vector < T > vctdiag(length * length, T(0.0));

    FORI (length) {
        vctdiag[i * length + i] = vct [i];
    }

    return vctdiag;
};


template <typename T>
vector< vector<T> > transposeVec ( const vector < vector < T > > & vMtx ) {
    assert( vMtx.size() != 0
            && vMtx[0].size() != 0 );

    Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic> m;
    m.resize(vMtx.size(), vMtx[0].size());

    FORIJ (m.rows(), m.cols()) m(i, j) = vMtx[i][j];
    m.transposeInPlace();

    vector < vector<T> > vTran( m.rows(), vector<T>(m.cols()) );
    FORIJ(m.rows(),  m.cols()) vTran[i][j] = m(i, j);

    return vTran;
};

template<typename T>
vector < T > xyToXYZ ( const vector < T > &xy )
{
    vector < T > XYZ (3);
    XYZ[0] = xy[0];
    XYZ[1] = xy[1];
    XYZ[2] = 1 - xy[0] - xy[1];

    return XYZ;
};

template<typename T>
vector < T > XYZTouv ( const vector < T > &XYZ )
{
    T uvS[] = { 4.0, 6.0 };
    T slice[] = { XYZ[0], XYZ[1] };
    vector < T > uvScale (uvS, uvS + sizeof(uvS)/sizeof(T));
    vector < T > vSlice (slice, slice + sizeof(slice)/sizeof(T));

    uvScale = mulVectorElement ( uvScale, vSlice );

    T scale = XYZ[0] + 15 * XYZ[1] + 3 * XYZ[2];
    scaleVector ( uvScale, 1.0 / scale);

    return uvScale;
};

template <typename T>
vector < vector < T > > diagVM ( const vector < T > & vct ) {
    assert( vct.size() != 0 );
    vector < vector<T> > vctdiag(vct.size(), vector<T>(vct.size(), T(0.0)));

    FORI(vct.size()) vctdiag[i][i] = vct[i];

    return vctdiag;
};

template<typename T>
vector < vector<T> > solveVM ( const vector < vector < T > > & vct1,
                               const vector < vector < T > > & vct2 ) {

    Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic> m1, m2, m3;
    m1.resize(vct1.size(), vct1[0].size());
    m2.resize(vct2.size(), vct2[0].size());

    FORIJ (vct1.size(), vct1[0].size())
        m1(i, j) = vct1[i][j];
    FORIJ (vct2.size(), vct2[0].size())
        m2(i, j) = vct2[i][j];

    // colPivHouseholderQr()
    m3 = m1.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(m2);

    vector < vector <T> > vct3 (m3.rows(), vector <T>(m3.cols()));
    FORIJ(m3.rows(), m3.cols()) vct3[i][j] = m3(i, j);

    return vct3;
};

template<typename T>
vector < vector < T > > getCAT ( const vector < T > & src,
                                 const vector < T > & des ) {
    assert(src.size() == des.size());

    vector < vector <T> > vcat(3, vector<T>(3));
    // cat02 or bradford
    FORIJ(3, 3) vcat[i][j] = cat02[i][j];

    vector< T > wSRC = mulVector ( src, vcat );
    vector< T > wDES = mulVector ( des, vcat );
    vector< vector<T> > vkm = solveVM(vcat, diagVM ( divVectorElement (wDES, wSRC) ));
    vkm = mulVector(vkm, transposeVec(vcat));

    return vkm;
}




template<typename T>
vector < T > uvToxy ( const vector < T > &uv )
{
    T xyS[] = { 3.0, 2.0 };
    vector < T > xyScale ( xyS, xyS + sizeof(xyS)/sizeof(T) );
    xyScale = mulVectorElement ( xyScale, uv );

    T scale = 2 * uv [0] - 8 * uv[1] + 4;
    scaleVector ( xyScale, 1.0 / scale );

    return xyScale;
};

template<typename T>
vector < T > uvToXYZ ( const vector < T > &uv )
{
    return xyToXYZ ( uvToxy ( uv ) );
};


DNGIdt::DNGIdt() {
	_cameraCalibration1DNG = vector < double > ( 9, 1.0 );
	_cameraCalibration2DNG = vector < double > ( 9, 1.0 );
	_cameraToXYZMtx        = vector < double > ( 9, 1.0 );
	_xyz2rgbMatrix1DNG     = vector < double > ( 9, 1.0 );
	_xyz2rgbMatrix2DNG     = vector < double > ( 9, 1.0 );
	_analogBalanceDNG      = vector < double > ( 3, 1.0 );
	_neutralRGBDNG         = vector < double > ( 3, 1.0 );
	_cameraXYZWhitePoint   = vector < double > ( 3, 1.0 );
	_calibrateIllum        = vector < double > ( 2, 1.0 );
	_baseExpo              = 1.0;
}

DNGIdt::DNGIdt ( libraw_rawdata_t R ) {
	_cameraCalibration1DNG = vector < double > ( 9, 1.0 );
	_cameraCalibration2DNG = vector < double > ( 9, 1.0 );
	_cameraToXYZMtx        = vector < double > ( 9, 1.0 );
	_xyz2rgbMatrix1DNG     = vector < double > ( 9, 1.0 );
	_xyz2rgbMatrix2DNG     = vector < double > ( 9, 1.0 );
	_analogBalanceDNG      = vector < double > ( 3, 1.0 );
	_neutralRGBDNG         = vector < double > ( 3, 1.0 );
	_cameraXYZWhitePoint   = vector < double > ( 3, 1.0 );
	_calibrateIllum        = vector < double > ( 2, 1.0 );

	_baseExpo = static_cast < double > ( R.color.baseline_exposure );
	_calibrateIllum[0] = static_cast < double > ( R.color.dng_color[0].illuminant );
	_calibrateIllum[1] = static_cast < double > ( R.color.dng_color[1].illuminant );

	FORI(3) {
		_neutralRGBDNG[i] = 1.0 / static_cast < double > ( R.color.cam_mul[i] );
	}

	FORIJ ( 3, 3 ) {
		_xyz2rgbMatrix1DNG[i*3+j] = static_cast < double > ( (R.color.dng_color[0].colormatrix)[i][j] );
		_xyz2rgbMatrix2DNG[i*3+j] = static_cast < double > ( (R.color.dng_color[1].colormatrix)[i][j] );
		_cameraCalibration1DNG[i*3+j] = static_cast < double > ( (R.color.dng_color[0].calibration)[i][j] );
		_cameraCalibration2DNG[i*3+j] = static_cast < double > ( (R.color.dng_color[1].calibration)[i][j] );
	}
}

DNGIdt::~DNGIdt() {
}

double DNGIdt::ccttoMired ( const double cct ) const {
	return 1.0E06 / cct;
}

double DNGIdt::robertsonLength ( const vector < double > & uv,
								 const vector < double > & uvt ) const {

	double t = uvt[2];
	vector < double > slope (2);
	slope[0] = -sign(t) / std::sqrt(1 + t * t);
	slope[1] = t * slope[0];

	vector < double > uvr ( uvt.begin(), uvt.begin()+2 );
	return cross2 ( slope, subVectors ( uv, uvr ) ) ;
}

double DNGIdt::lightSourceToColorTemp ( const unsigned short tag ) const {

	if ( tag >= 32768 )
		return ( static_cast < double > ( tag ) ) - 32768.0;

	uint16_t LightSourceEXIFTagValues[][2] = {
		{ 0,    5500 },
		{ 1,    5500 },
		{ 2,    3500 },
		{ 3,    3400 },
		{ 10,   5550 },
		{ 17,   2856 },
		{ 18,   4874 },
		{ 19,   6774 },
		{ 20,   5500 },
		{ 21,   6500 },
		{ 22,   7500 }
	};

	FORI ( countSize ( LightSourceEXIFTagValues ) ) {
		if ( LightSourceEXIFTagValues[i][0] == static_cast < uint16_t > (tag) ) {
			return ( static_cast < double > (LightSourceEXIFTagValues[i][1]) );
		}
	}

	return 5500.0;
}

double DNGIdt::XYZToColorTemperature ( const vector < double > & XYZ ) const {

	vector < double > uv = XYZTouv ( XYZ );
	int Nrobert = countSize ( Robertson_uvtTable );
	int i;

	double mired;
	double RDthis = 0.0, RDprevious = 0.0;

	for ( i = 0; i < Nrobert; i++ ) {
		vector < double > robertson ( Robertson_uvtTable[i],
									  Robertson_uvtTable[i] + countSize(Robertson_uvtTable[i]) );
		if (( RDthis = robertsonLength ( uv, robertson ) ) <= 0.0 )
			break;
		RDprevious = RDthis;
	}
	if ( i <= 0 )
		mired = RobertsonMired[0];
	else if ( i >= Nrobert )
		mired = RobertsonMired[Nrobert - 1];
	else
		mired = RobertsonMired[i-1] + RDprevious * ( \
				RobertsonMired[i] - RobertsonMired[i-1] )\
				/( RDprevious-RDthis );

	double cct = 1.0e06 / mired;
	cct = std::max ( 2000.0, std::min ( 50000.0, cct ) );

	return cct;
}

vector < double > DNGIdt::XYZtoCameraWeightedMatrix ( const double & mir0,
													  const double & mir1,
													  const double & mir2 ) const {

	double weight = std::max ( 0.0, std::min ( 1.0, (mir1 - mir0) / (mir1 - mir2) ) );
	vector < double > result = subVectors ( _xyz2rgbMatrix2DNG, _xyz2rgbMatrix1DNG );
	scaleVector ( result, weight );
	result = addVectors ( result, _xyz2rgbMatrix1DNG );

	return result;
}

vector < double > DNGIdt::findXYZtoCameraMtx ( const vector < double > & neutralRGB ) const {

	if ( _calibrateIllum.size() == 0 ) {
		fprintf ( stderr, " No calibration illuminants were found. \n " );
		return _xyz2rgbMatrix1DNG;
	}

	if ( neutralRGB.size() == 0 ) {
		fprintf ( stderr, " no neutral RGB values were found. \n " );
		return _xyz2rgbMatrix1DNG;
	}

	double cct1 = lightSourceToColorTemp ( static_cast < const unsigned short > ( _calibrateIllum[0] ) );
	double cct2 = lightSourceToColorTemp ( static_cast < const unsigned short > ( _calibrateIllum[1] ) );

	double mir1 = ccttoMired ( cct1 );
	double mir2 = ccttoMired ( cct2 );

	double maxMir = ccttoMired ( 2000.0 );
	double minMir = ccttoMired ( 50000.0 );

	double lomir = std::max ( minMir, std::min ( maxMir, std::min ( mir1, mir2 ) ) );
	double himir = std::max ( minMir, std::min ( maxMir, std::max ( mir1, mir2 ) ) );
	double mirStep = std::max ( 5.0, ( himir - lomir ) / 50.0 );

	double mir = 0.0, lastMired = 0.0, estimatedMired = 0.0, lerror = 0.0, lastError = 0.0, smallestError = 0.0;

	for ( mir = lomir; mir < himir;  mir += mirStep ) {
		lerror = mir - ccttoMired ( XYZToColorTemperature ( mulVector \
								   ( invertV ( XYZtoCameraWeightedMatrix ( mir, mir1, mir2 ) ),\
									_neutralRGBDNG ) ) );

		if ( std::fabs( lerror - 0.0 ) <= 1e-09 ) {
			estimatedMired = mir;
			break;
		}
		if ( std::fabs( mir - lomir - 0.0 ) > 1e-09
			 && lerror * lastError <= 0.0 ) {
			estimatedMired = mir + ( lerror / ( lerror-lastError ) * ( mir - lastMired ) );
			break;
		}
		if ( std::fabs( mir - lomir ) <= 1e-09
			 || std::fabs ( lerror ) < std::fabs ( smallestError ) )    {
			estimatedMired = mir ;
			smallestError = lerror;
		}

		lastError = lerror;
		lastMired = mir;
	}

	return XYZtoCameraWeightedMatrix ( estimatedMired, mir1, mir2 );
}

vector < double > DNGIdt::colorTemperatureToXYZ ( const double & cct ) const {

	double mired = 1.0e06 / cct;
	vector < double > uv ( 2, 1.0 );

	int Nrobert = countSize (Robertson_uvtTable);
	int i;

	for ( i = 0; i < Nrobert; i++ ) {
		if ( RobertsonMired[i] >= mired )
			break;
	}

	if ( i <= 0 ) {
		uv = vector < double > ( Robertson_uvtTable[0], Robertson_uvtTable[0] + 2 );
	}
	else if ( i >= Nrobert ) {
		uv = vector < double > ( Robertson_uvtTable[Nrobert - 1], Robertson_uvtTable[Nrobert - 1] + 2 );
	}
	else {
		double weight = ( mired - RobertsonMired[i-1] ) / ( RobertsonMired[i] - RobertsonMired[i-1] );

		vector < double > uv1 ( Robertson_uvtTable[i], Robertson_uvtTable[i] + 2 );
		scaleVector ( uv1, weight );

		vector < double > uv2 ( Robertson_uvtTable[i-1], Robertson_uvtTable[i-1] + 2 );
		scaleVector ( uv2, 1.0 - weight );

		uv = addVectors ( uv1, uv2 );
	}

	return uvToXYZ(uv);
}

vector < double > DNGIdt::matrixRGBtoXYZ ( const double chromaticities[][2] ) const {
	vector < double > rXYZ = xyToXYZ ( vector< double > ( chromaticities[0], chromaticities[0] + 2 ) );
	vector < double > gXYZ = xyToXYZ ( vector< double > ( chromaticities[1], chromaticities[1] + 2 ) );
	vector < double > bXYZ = xyToXYZ ( vector< double > ( chromaticities[2], chromaticities[2] + 2 ) );
	vector < double > wXYZ = xyToXYZ ( vector< double > ( chromaticities[3], chromaticities[3] + 2 ) );

	vector < double > rgbMtx(9);
	FORI(3) {
		rgbMtx[0+i*3] = rXYZ[i];
		rgbMtx[1+i*3] = gXYZ[i];
		rgbMtx[2+i*3] = bXYZ[i];
	}

	scaleVector ( wXYZ, 1.0 / wXYZ[1] );

	vector < double > channelgains = mulVector ( invertV ( rgbMtx ), wXYZ, 3 );
	vector < double > colorMatrix  = mulVector ( rgbMtx, diagV ( channelgains ), 3 );

	return colorMatrix;
}

void DNGIdt::getCameraXYZMtxAndWhitePoint ( ) {
	_cameraToXYZMtx = invertV ( findXYZtoCameraMtx ( _neutralRGBDNG ) );
	assert ( std::fabs ( sumVector ( _cameraToXYZMtx ) - 0.0 ) > 1e-09 );

	scaleVector ( _cameraToXYZMtx, std::pow ( 2.0, _baseExpo ) );

	if ( _neutralRGBDNG.size() > 0 ) {
		_cameraXYZWhitePoint = mulVector ( _cameraToXYZMtx, _neutralRGBDNG );
	} else {
		_cameraXYZWhitePoint = colorTemperatureToXYZ ( lightSourceToColorTemp ( _calibrateIllum[0] ) );
	}

	scaleVector ( _cameraXYZWhitePoint, 1.0 / _cameraXYZWhitePoint[1] );
	assert ( sumVector ( _cameraXYZWhitePoint ) != 0);

	return;
}

vector < vector < double > > DNGIdt::getDNGCATMatrix ( ) {
	vector < double > deviceWhiteV ( 3, 1.0 );
	getCameraXYZMtxAndWhitePoint ( );
	vector < double > outputRGBtoXYZMtx = matrixRGBtoXYZ ( chromaticitiesACES );
	vector < double > outputXYZWhitePoint = mulVector ( outputRGBtoXYZMtx, deviceWhiteV );
	vector < vector < double > > chadMtx = getCAT ( _cameraXYZWhitePoint, outputXYZWhitePoint );

	return chadMtx;
}

vector < vector < double > > DNGIdt::getDNGIDTMatrix ( ) {
	vector < vector < double > > chadMtx = getDNGCATMatrix ( );
	vector < double > XYZ_acesrgb (9), CAT (9), ap0toap1(9);
	FORIJ ( 3, 3 ) {
		XYZ_acesrgb[i*3+j] = XYZ_acesrgb_3[i][j];
		CAT[i*3+j] = chadMtx[i][j];
	}

	vector < double > matrix = mulVector ( XYZ_acesrgb, CAT, 3 );
	vector < vector < double > > DNGIDTMatrix ( 3, vector < double > (3) );
	FORIJ ( 3, 3 ) DNGIDTMatrix[i][j] = matrix[i*3+j];

	assert ( std::fabs( sumVectorM ( DNGIDTMatrix ) - 0.0 ) > 1e-09 );

	return DNGIDTMatrix;
}

void DNGIdt::getDNGIDTMatrix2 (float* mat)
{
	vector < vector < double > > chadMtx = getDNGCATMatrix ( );
	vector < double > XYZ_acesrgb (9), CAT (9);
	FORIJ ( 3, 3 ) {
		XYZ_acesrgb[i*3+j] = XYZ_acesrgb_3[i][j];
		CAT[i*3+j] = chadMtx[i][j];
	}

	vector < double > matrix = mulVector ( XYZ_acesrgb, CAT, 3 );
	vector < vector < double > > DNGIDTMatrix ( 3, vector < double > (3) );
	FORIJ ( 3, 3 ) DNGIDTMatrix[i][j] = matrix[i*3+j];

	assert ( std::fabs( sumVectorM ( DNGIDTMatrix ) - 0.0 ) > 1e-09 );

	FORI(9) mat[i] = matrix[i];
}
