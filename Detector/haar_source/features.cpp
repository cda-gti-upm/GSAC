#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "traincascade_features.h"
#include "haarfeatures.h"
//#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

#define CC_CASCADE_FILENAME "cascade.xml"
#define CC_PARAMS_FILENAME "params.xml"

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE "stageType"
#define CC_FEATURE_TYPE "featureType"
#define CC_HEIGHT "height"
#define CC_WIDTH  "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_BOOST_TYPE       "boostType"
#define CC_DISCRETE_BOOST   "DAB"
#define CC_REAL_BOOST       "RAB"
#define CC_LOGIT_BOOST      "LB"
#define CC_GENTLE_BOOST     "GAB"
#define CC_MINHITRATE       "minHitRate"
#define CC_MAXFALSEALARM    "maxFalseAlarm"
#define CC_TRIM_RATE        "weightTrimRate"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       FEATURES
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"
#define CC_FEATURE_SIZE   "featSize"

#define CC_HAAR        "HAAR"
#define CC_MODE        "mode"
#define CC_MODE_BASIC  "BASIC"
#define CC_MODE_CORE   "CORE"
#define CC_MODE_ALL    "ALL"
#define CC_RECTS       "rects"
#define CC_TILTED      "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_HOG "HOG"

float calcNormFactor( const Mat& sum, const Mat& sqSum )
{
    CV_DbgAssert( sum.cols > 3 && sqSum.rows > 3 );
    Rect normrect( 1, 1, sum.cols - 3, sum.rows - 3 );
    size_t p0, p1, p2, p3;
    CV_SUM_OFFSETS( p0, p1, p2, p3, normrect, sum.step1() )
    double area = normrect.width * normrect.height;
    const int *sp = (const int*)sum.data;
    int valSum = sp[p0] - sp[p1] - sp[p2] + sp[p3];
    const double *sqp = (const double *)sqSum.data;
    double valSqSum = sqp[p0] - sqp[p1] - sqp[p2] + sqp[p3];
    return (float) sqrt( (double) (area * valSqSum - (double)valSum * valSum) );
}

CvParams::CvParams() : name( "params" ) {}
void CvParams::printDefaults() const
{ cout << "--" << name << "--" << endl; }
void CvParams::printAttrs() const {}
bool CvParams::scanAttr( const string, const string ) { return false; }


//---------------------------- FeatureParams --------------------------------------

CvFeatureParams::CvFeatureParams() : maxCatCount( 0 ), featSize( 1 )
{
    name = CC_FEATURE_PARAMS;
}

void CvFeatureParams::init( const CvFeatureParams& fp )
{
    maxCatCount = fp.maxCatCount;
    featSize = fp.featSize;
}

void CvFeatureParams::write( FileStorage &fs ) const
{
    fs << CC_MAX_CAT_COUNT << maxCatCount;
    fs << CC_FEATURE_SIZE << featSize;
}

bool CvFeatureParams::read( const FileNode &node )
{
    if ( node.empty() )
        return false;
    maxCatCount = node[CC_MAX_CAT_COUNT];
    featSize = node[CC_FEATURE_SIZE];
    return ( maxCatCount >= 0 && featSize >= 1 );
}

Ptr<CvFeatureParams> CvFeatureParams::create( int featureType )
{
    return featureType == HAAR ? Ptr<CvFeatureParams>(new CvHaarFeatureParams) :
        Ptr<CvFeatureParams>();
}

//------------------------------------- FeatureEvaluator ---------------------------------------

void CvFeatureEvaluator::init(const CvFeatureParams *_featureParams,
                              int _maxSampleCount, Size _winSize )
{
    CV_Assert(_maxSampleCount > 0);
    featureParams = (CvFeatureParams *)_featureParams;
    winSize = _winSize;
    numFeatures = 0;
    cls.create( (int)_maxSampleCount, 1, CV_32FC1 );
    generateFeatures();
}

void CvFeatureEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_Assert(img.cols == winSize.width);
    CV_Assert(img.rows == winSize.height);
    CV_Assert(idx < cls.rows);
    cls.ptr<float>(idx)[0] = clsLabel;
}

Ptr<CvFeatureEvaluator> CvFeatureEvaluator::create(int type)
{
    return type == CvFeatureParams::HAAR ? Ptr<CvFeatureEvaluator>(new CvHaarEvaluator) :
        Ptr<CvFeatureEvaluator>();
}
