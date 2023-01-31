#include <storm/storage/TriangularFuzzyNumber.h>
#include <storm/storage/FuzzyAnalysisResult.h>
#include <storm/adapters/RationalFunctionAdapter.h>
#include <vector>
#include "benchmark.cpp"
#include <float.h>
#include <limits>

storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> generateRandomFuzzyMatrix(unsigned int size)
{
    storm::storage::SparseMatrixBuilder<storm::storage::TriangularFuzzyNumber> matrixBuilder;
    std::vector<std::vector<storm::storage::TriangularFuzzyNumber>> values(size, std::vector<storm::storage::TriangularFuzzyNumber>(size));

    for (unsigned int row = 0; row < size; ++row)
    {
        double rowSum = 0;
        for (unsigned int col = 0; col < size; ++col)
        {
            double lower, middle, upper;
            lower = 0.0;
            middle = storm::storage::randomDouble(0.001, 0.999);
            upper = 1.0;
            rowSum += middle;
            values[row][col] = storm::storage::TriangularFuzzyNumber(lower, middle, upper);
        }
        if (rowSum != 1)
        {
            double factor = 1.0 / rowSum;
            for (unsigned int col = 0; col < size; ++col)
            {
                double new_middle = values[row][col].getPeak() * factor;
                values[row][col] = storm::storage::TriangularFuzzyNumber(storm::storage::randomDouble(0.0, new_middle), new_middle, storm::storage::randomDouble(new_middle, 1.0));
            }
        }
    }
    for (unsigned int row = 0; row < size; ++row)
    {
        for (unsigned int col = 0; col < size; ++col)
        {
            matrixBuilder.addNextValue(row, col, values[row][col]);
        }
    }
    return matrixBuilder.build();
}

void print2DVector(const std::vector<std::vector<double>> &vec)
{
    for (const auto &row : vec)
    {
        for (const auto &element : row)
        {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 3x3 matrix
storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> createFuzzyMatrix3()
{
    storm::storage::SparseMatrixBuilder<storm::storage::TriangularFuzzyNumber> matrixBuilder;

    matrixBuilder.addNextValue(0, 1, storm::storage::TriangularFuzzyNumber(0.03, 0.32, 0.88));
    matrixBuilder.addNextValue(0, 2, storm::storage::TriangularFuzzyNumber(0.65, 0.68, 0.83));

    matrixBuilder.addNextValue(1, 1, storm::storage::TriangularFuzzyNumber(0, 1.0, 0));

    matrixBuilder.addNextValue(2, 0, storm::storage::TriangularFuzzyNumber(0.5, 0.61, 0.98));
    matrixBuilder.addNextValue(2, 2, storm::storage::TriangularFuzzyNumber(0.08, 0.39, 0.46));

    return matrixBuilder.build();
}

// 4x4 matrix
storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> createFuzzyMatrix4()
{
    storm::storage::SparseMatrixBuilder<storm::storage::TriangularFuzzyNumber> matrixBuilder;

    matrixBuilder.addNextValue(0, 2, storm::storage::TriangularFuzzyNumber(0.0, 1.0, 0.0));

    matrixBuilder.addNextValue(1, 0, storm::storage::TriangularFuzzyNumber(0.02, 0.05, 0.53));
    matrixBuilder.addNextValue(1, 2, storm::storage::TriangularFuzzyNumber(0.23, 0.34, 0.53));
    matrixBuilder.addNextValue(1, 3, storm::storage::TriangularFuzzyNumber(0.51, 0.61, 0.94));

    matrixBuilder.addNextValue(2, 0, storm::storage::TriangularFuzzyNumber(0, 1.0, 0));

    matrixBuilder.addNextValue(3, 0, storm::storage::TriangularFuzzyNumber(0.17, 0.51, 0.54));
    matrixBuilder.addNextValue(3, 1, storm::storage::TriangularFuzzyNumber(0.09, 0.49, 0.5));

    return matrixBuilder.build();
}

// 4x4 matrix
storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> createFuzzyMatrix4alt()
{
    storm::storage::SparseMatrixBuilder<storm::storage::TriangularFuzzyNumber> matrixBuilder;

    matrixBuilder.addNextValue(0, 1, storm::storage::TriangularFuzzyNumber(0.01, 0.23, 1.0));
    matrixBuilder.addNextValue(0, 3, storm::storage::TriangularFuzzyNumber(0.42, 0.77, 0.79));

    matrixBuilder.addNextValue(1, 0, storm::storage::TriangularFuzzyNumber(0.33, 0.79, 0.9));
    matrixBuilder.addNextValue(1, 2, storm::storage::TriangularFuzzyNumber(0.15, 0.21, 0.4));

    matrixBuilder.addNextValue(2, 0, storm::storage::TriangularFuzzyNumber(0.19, 0.24, 1.0));
    matrixBuilder.addNextValue(2, 1, storm::storage::TriangularFuzzyNumber(0.27, 0.76, 0.98));

    matrixBuilder.addNextValue(3, 1, storm::storage::TriangularFuzzyNumber(0.0, 1.0, 0.0));

    return matrixBuilder.build();
}

// 5x5 matrix
storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> createFuzzyMatrix5()
{
    storm::storage::SparseMatrixBuilder<storm::storage::TriangularFuzzyNumber> matrixBuilder;

    matrixBuilder.addNextValue(0, 1, storm::storage::TriangularFuzzyNumber(0.42, 0.44, 0.53));
    matrixBuilder.addNextValue(0, 2, storm::storage::TriangularFuzzyNumber(0.29, 0.56, 0.74));

    matrixBuilder.addNextValue(1, 4, storm::storage::TriangularFuzzyNumber(0, 1.0, 0));

    matrixBuilder.addNextValue(2, 0, storm::storage::TriangularFuzzyNumber(0.21, 0.3, 0.55));
    matrixBuilder.addNextValue(2, 3, storm::storage::TriangularFuzzyNumber(0.18, 0.66, 0.87));
    matrixBuilder.addNextValue(2, 4, storm::storage::TriangularFuzzyNumber(0.03, 0.04, 0.43));

    matrixBuilder.addNextValue(3, 4, storm::storage::TriangularFuzzyNumber(0, 1.0, 0));

    matrixBuilder.addNextValue(4, 0, storm::storage::TriangularFuzzyNumber(0.29, 0.69, 0.79));
    matrixBuilder.addNextValue(4, 2, storm::storage::TriangularFuzzyNumber(0.02, 0.05, 0.58));
    matrixBuilder.addNextValue(4, 3, storm::storage::TriangularFuzzyNumber(0.13, 0.26, 0.62));

    return matrixBuilder.build();
}

// alpha=.84 alpha-cut for the 3x3 matrix
std::vector<std::vector<storm::Interval>> createIntervalMatrix3()
{
    std::vector<std::vector<storm::Interval>> v;
    v.push_back({storm::Interval(0), storm::Interval(0.2736, 0.4096), storm::Interval(0.6752, 0.704)});
    v.push_back({storm::Interval(0), storm::Interval(1), storm::Interval(0)});
    v.push_back({storm::Interval(0.5924, 0.6692), storm::Interval(0), storm::Interval(0.3404, 0.4012)});

    return v;
}

// crisp (INVALID) matrix out of alpha=.84 alpha-cut for the 3x3 matrix
std::vector<std::vector<double>> createCrispMatrix3()
{
    std::vector<std::vector<double>> v;
    v.push_back({0, 0.2907, 0.6873});
    v.push_back({0, 1, 0});
    v.push_back({0.6099, 0, 0.3756});

    return v;
}

// alpha=.23 alpha-cut for the 4x4 matrix
std::vector<std::vector<storm::Interval>> createIntervalMatrix4()
{

    std::vector<std::vector<storm::Interval>> v;
    v.push_back({storm::Interval(0), storm::Interval(0), storm::Interval(1), storm::Interval(0)});
    v.push_back({storm::Interval(0.0269, 0.4196), storm::Interval(0), storm::Interval(0.2553, 0.4863), storm::Interval(0.533, 0.8641)});
    v.push_back({storm::Interval(1), storm::Interval(0), storm::Interval(0), storm::Interval(0)});
    v.push_back({storm::Interval(0.2482, 0.5331), storm::Interval(0.182, 0.4977), storm::Interval(0), storm::Interval(0)});

    return v;
}

// crisp(INVALID) matrix out of alpha=.23 alpha-cut for the 4x4 matrix
std::vector<std::vector<double>> createCrispMatrix4()
{

    std::vector<std::vector<double>> v;
    v.push_back({0, 0, 1, 0});
    v.push_back({0.1584, 0, 0.3189, 0.6847});
    v.push_back({1, 0, 0, 0});
    v.push_back({0.3473, 0.4944, 0, 0});

    return v;
}
// alpha=.82 alpha-cut for the 5x5 matrix
std::vector<std::vector<storm::Interval>> createIntervalMatrix5a()
{
    std::vector<std::vector<storm::Interval>> v;
    v.push_back({storm::Interval(0), storm::Interval(0.4364, 0.4562), storm::Interval(0.5114, 0.5924), storm::Interval(0), storm::Interval(0)});
    v.push_back({storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(1)});
    v.push_back({storm::Interval(0.2838, 0.345), storm::Interval(0), storm::Interval(0), storm::Interval(0.5736, 0.6978), storm::Interval(0.0382, 0.1102)});
    v.push_back({storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(1)});
    v.push_back({storm::Interval(0.618, 0.708), storm::Interval(0), storm::Interval(0.0446, 0.1454), storm::Interval(0.2366, 0.3248), storm::Interval(0)});

    return v;
}

// alpha=.53 alpha-cut for the 5x5 matrix
std::vector<std::vector<storm::Interval>> createIntervalMatrix5b()
{
    std::vector<std::vector<storm::Interval>> v;
    v.push_back({storm::Interval(0), storm::Interval(0.4306, 0.4823), storm::Interval(0.4331, 0.6446), storm::Interval(0), storm::Interval(0)});
    v.push_back({storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(1)});
    v.push_back({storm::Interval(0.2577, 0.4175), storm::Interval(0), storm::Interval(0), storm::Interval(0.4344, 0.7587), storm::Interval(0.0353, 0.2233)});
    v.push_back({storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(0), storm::Interval(1)});
    v.push_back({storm::Interval(0.502, 0.737), storm::Interval(0), storm::Interval(0.0359, 0.2991), storm::Interval(0.1989, 0.4292), storm::Interval(0)});

    return v;
}

// crisp (INVALID) matrix #1 out of alpha=.53 alpha-cut for the 5x5 matrix
std::vector<std::vector<double>> createCrispMatrix5()
{
    std::vector<std::vector<double>> v;
    v.push_back({0, 0.4811, 0.5486, 0, 0});
    v.push_back({0, 0, 0, 0, 1});
    v.push_back({0.3384, 0, 0, 0.7256, 0.2177});
    v.push_back({0, 0, 0, 0, 1});
    v.push_back({0.588, 0, 0.1627, 0.3011, 0});

    return v;
}

// crisp (INVALID) matrix #2 out of alpha=.53 alpha-cut for the 5x5 matrix
std::vector<std::vector<double>> createCrispMatrix5alt()
{
    std::vector<std::vector<double>> v;
    v.push_back({0, 0.4651, 0.4761, 0, 0});
    v.push_back({0, 0, 0, 0, 1});
    v.push_back({0.3743, 0, 0, 0.5137, 0.0825});
    v.push_back({0, 0, 0, 0, 1});
    v.push_back({0.588, 0, 0.2775, 0.3248, 0});

    return v;
}

bool testMatrixMul(std::vector<std::vector<double>> testcase, std::vector<std::vector<double>> expected, int power)
{
    storm::storage::FuzzyAnalysisResult::member m(testcase, power, {0, 0});
    std::vector<std::vector<double>> result = storm::storage::matrixMul(m.getMatrix(), m.getN());

    for (int i = 0; i < result.size(); i++)
    {
        for (int j = 0; j < result[i].size(); j++)
        {
            if (std::abs(result[i][j] - expected[i][j]) >= 0.0001)
            {
                std::cout << result[i][j] << " =/= " << expected[i][j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool testGetIntervalMatrix(storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> testcase, std::pair<std::vector<std::vector<storm::Interval>>, std::vector<std::pair<int, int>>> expected, double alpha)
{
    storm::storage::FuzzyAnalysisResult test(testcase);
    auto result1 = test.getIntervalMatrix(alpha).first;
    auto result2 = test.getIntervalMatrix(alpha).second;

    for (int i = 0; i < result1.size(); i++)
    {
        for (int j = 0; j < result1[i].size(); j++)
        {
            if (std::abs(result1[i][j].lower() - expected.first[i][j].lower()) >= 0.0001)
            {
                std::cout << result1[i][j].lower() << " =/= " << expected.first[i][j].lower() << std::endl;
                return false;
            }
            if (std::abs(result1[i][j].upper() - expected.first[i][j].upper()) >= 0.0001)
            {
                std::cout << result1[i][j].upper() << " =/= " << expected.first[i][j].upper() << std::endl;
                return false;
            }
        }
    }

    for (int i = 0; i < result2.size(); i++)
    {
        if (result2[i].first - expected.second[i].first)
        {
            std::cout << result2[i].first << " =/= " << expected.second[i].first << std::endl;
            return false;
        }
        if (result2[i].second - expected.second[i].second)
        {
            std::cout << result2[i].second << " =/= " << expected.second[i].second << std::endl;
            return false;
        }
    }

    return true;
}

bool testGetAlphaCut(storm::storage::TriangularFuzzyNumber testcase, storm::Interval expected, double alpha)
{
    storm::storage::FuzzyAnalysisResult test(createFuzzyMatrix3());
    auto result = test.getAlphaCut(testcase, alpha);

    if (std::abs(result.lower() - expected.lower()) >= 0.0001)
    {
        std::cout << result.lower() << " =/= " << expected.lower() << std::endl;
        return false;
    }
    if (std::abs(result.upper() - expected.upper()) >= 0.0001)
    {
        std::cout << result.upper() << " =/= " << expected.upper() << std::endl;
        return false;
    }
    return true;
}

bool testGetFeasibleMutationRange(double mutationValue1, double mutationValue2, storm::Interval mutationRange1, storm::Interval mutationRange2, storm::Interval expected)
{
    auto result = storm::storage::getFeasibleMutationRange(mutationValue1, mutationValue2, mutationRange1, mutationRange2);

    if (std::abs(result.lower() - expected.lower()) >= 0.0001)
    {
        std::cout << result.lower() << " =/= " << expected.lower() << std::endl;
        return false;
    }
    if (std::abs(result.upper() - expected.upper()) >= 0.0001)
    {
        std::cout << result.upper() << " =/= " << expected.upper() << std::endl;
        return false;
    }
    return true;
}

void integrationTest()
{
    auto fuzzyMatrix4 = createFuzzyMatrix4();
    storm::storage::FuzzyAnalysisResult test(fuzzyMatrix4);

    std::pair<std::vector<std::vector<storm::Interval>>, std::vector<std::pair<int, int>>> matrixAndIndexes = test.getIntervalMatrix(0.5);
    int populationSize = 10;
    int steps = 5;
    std::pair<int, int> idx = {1, 3};
    std::vector<storm::storage::FuzzyAnalysisResult::member> population = test.initializePopulation(matrixAndIndexes.first, populationSize, steps, idx);
    std::cout << "INITIAL POPULATION" << std::endl;
    for (auto &member : population)
    {
        print2DVector(member.getMatrix());
        std::cout << std::endl;
    }

    double selectionSample = 0.3;
    population = test.selectPopulation(population, selectionSample, true);
    std::cout << "SELECTED POPULATION" << std::endl;
    for (auto &member : population)
    {
        print2DVector(member.getMatrix());
        std::cout << std::endl;
        std::cout << member.getFitness() << std::endl;
    }

    std::vector<int> intervalRowIndices = {1, 3};
    population = test.crossPopulation(population, populationSize, intervalRowIndices);
    std::cout << "CROSSED POPULATION" << std::endl;
    for (auto &member : population)
    {
        print2DVector(member.getMatrix());
        std::cout << std::endl;
    }

    double mutationRate = 0.2;
    std::vector<std::pair<int, int>> intervalIndices = {{1, 0}, {1, 2}, {1, 3}, {3, 0}, {3, 1}};
    population = test.mutatePopulation(population, mutationRate, intervalIndices, intervalRowIndices, matrixAndIndexes.first);
    std::cout << "MUTATED POPULATION" << std::endl;
    for (auto &member : population)
    {
        print2DVector(member.getMatrix());
        std::cout << std::endl;
    }
}

double averageMiddleValues(std::vector<double> values)
{

    double sum = 0;
    double min = *std::min_element(values.begin(), values.end());
    double max = *std::max_element(values.begin(), values.end());
    for (int i = 0; i < values.size(); i++)
    {
        if (values[i] != min && values[i] != max)
            sum += values[i];
    }

    return sum / (values.size() - 2);
}

void benchmark(storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> fuzzyMatrix, int steps, std::pair<int, int> idx, bool isMin, double alpha, int milliseconds)
{
    // initial parameter values
    int populationSize = 50;
    double selectionSample = 0.05;
    double mutationRate = 0.05;
    std::vector<std::vector<double>> best_parameter_combs;
    int generations;
    double current_result;
    double best_result = isMin ? std::numeric_limits<double>::max() : std::numeric_limits<double>::min();
    int sample = 5;

    // FAR preparation
    storm::storage::FuzzyAnalysisResult analysis = storm::storage::FuzzyAnalysisResult(fuzzyMatrix);
    std::pair<std::vector<std::vector<storm::Interval>>, std::vector<std::pair<int, int>>> matrixAndIndexes = analysis.getIntervalMatrix(alpha);
    std::vector<std::vector<storm::Interval>> intervalMatrix = matrixAndIndexes.first;
    std::vector<std::pair<int, int>> intervalIndices = matrixAndIndexes.second;
    std::vector<int> intervalRowIndices;
    std::transform(intervalIndices.begin(), intervalIndices.end(), std::back_inserter(intervalRowIndices),
                   [](const std::pair<int, int> &p)
                   { return p.first; });

    // logging
    std::ofstream log_file;
    std::string file_name = std::to_string(intervalMatrix.size()) + "-" + std::to_string(milliseconds) + ".txt";
    log_file.open(file_name, std::ofstream::out | std::ofstream::trunc);

    // search for minima of parameters
    log_file << "Begin for size: " << intervalMatrix.size() << ", ms: " << milliseconds << std::endl;
    for (int i = populationSize; i <= 200; i += 50)
    {
        for (double j = selectionSample; j <= 0.2; j += 0.05)
        {
            for (double k = mutationRate; k <= 0.2; k += 0.05)
            {
                std::vector<double> results;
                log_file << "Checking " << i << " " << j << " " << k << std::endl;
                for (int l = 0; l < sample; l++)
                {
                    results.push_back(analysis.timeBasedMatrixMul(intervalMatrix, intervalIndices, steps, idx, isMin, i, milliseconds, j, k));
                }
                current_result = averageMiddleValues(results);
                if ((isMin && current_result < best_result) || (!isMin && current_result > best_result))
                {
                    log_file << "BEST" << std::endl;
                    log_file << best_result << " " << current_result << std::endl;
                    best_result = current_result;
                    best_parameter_combs.clear();
                    best_parameter_combs.push_back({(double)i, j, k});
                }
                else if ((isMin && current_result == best_result) || (!isMin && current_result == best_result))
                {
                    log_file << "EQUAL" << std::endl;
                    log_file << best_result << " " << current_result << std::endl;
                    best_parameter_combs.push_back({(double)i, j, k});
                }
                else
                {
                    log_file << "WORSE" << std::endl;
                    log_file << best_result << " " << current_result << std::endl;
                }
            }
        }
    }
    log_file << best_result << std::endl;
    print2DVector(best_parameter_combs);
    log_file.close();
}

void run_benchmarks()
{
    // auto matrix10_1 = generateRandomFuzzyMatrix(10);
    // benchmark(matrix10_1, 3, {9, 3}, true, 0.5, 1000);
    // auto matrix10_2 = generateRandomFuzzyMatrix(10);
    // benchmark(matrix10_2, 3, {6, 7}, true, 0.5, 9000);
    // auto matrix10_3 = generateRandomFuzzyMatrix(10);
    // benchmark(matrix10_3, 3, {8, 4}, true, 0.5, 25000);

    // auto matrix30_1 = generateRandomFuzzyMatrix(30);
    // benchmark(matrix30_1, 3, {0, 14}, true, 0.5, 9000);
    auto matrix30_2 = generateRandomFuzzyMatrix(30);
    benchmark(matrix30_2, 3, {11, 9}, true, 0.5, 25000);
    // auto matrix30_3 = generateRandomFuzzyMatrix(30);
    // benchmark(matrix30_3, 3, {7, 4}, true, 0.5, 49000);

    auto matrix50_1 = generateRandomFuzzyMatrix(50);
    benchmark(matrix50_1, 3, {24, 9}, true, 0.5, 25000);
    // auto matrix50_2 = generateRandomFuzzyMatrix(50);
    // benchmark(matrix50_2, 3, {36, 11}, true, 0.5, 49000);
    // auto matrix50_3 = generateRandomFuzzyMatrix(50);
    // benchmark(matrix50_3, 3, {4, 21}, true, 0.5, 81000);
}

void runTests()
{
    // PREP
    auto fuzzyMatrix3 = createFuzzyMatrix3();
    auto fuzzyMatrix4 = createFuzzyMatrix4();
    auto fuzzyMatrix5 = createFuzzyMatrix5();

    auto intervalMatrix3 = createIntervalMatrix3();
    auto intervalMatrix4 = createIntervalMatrix4();
    auto intervalMatrix5a = createIntervalMatrix5a();
    auto intervalMatrix5b = createIntervalMatrix5b();

    auto crispMatrix3 = createCrispMatrix3();
    auto crispMatrix4 = createCrispMatrix4();
    auto crispMatrix5 = createCrispMatrix5();
    auto crispMatrix5alt = createCrispMatrix5alt();

    // testMatrixMul
    std::cout << testMatrixMul(crispMatrix3, {{0.1233, 0.6168, 0.1834}, {0, 1, 0}, {0.1627, 0.5267, 0.2236}}, 7) << std::endl;
    std::cout << testMatrixMul(crispMatrix4, {{0, 0, 1, 0}, {0.2391, 0, 0.8392, 0.003}, {1, 0, 0, 0}, {0.7615, 0.0021, 0.1178, 0}}, 11) << std::endl;
    std::cout << testMatrixMul(crispMatrix5, {{0.5351, 0.2411, 0.3891, 0.4744, 0.7481}, {0.4914, 0.2216, 0.3574, 0.4366, 0.6897}, {0.6109, 0.2749, 0.4436, 0.5423, 0.8540}, {0.4914, 0.2216, 0.3574, 0.4366, 0.6897}, {0.5264, 0.2364, 0.3818, 0.4670, 0.7361}}, 13) << std::endl;

    // testGetIntervalMatrix
    std::cout << testGetIntervalMatrix(fuzzyMatrix3, {intervalMatrix3, {{0, 1}, {0, 2}, {2, 0}, {2, 2}}}, 0.84) << std::endl;
    std::cout << testGetIntervalMatrix(fuzzyMatrix4, {intervalMatrix4, {{1, 0}, {1, 2}, {1, 3}, {3, 0}, {3, 1}}}, 0.23) << std::endl;
    std::cout << testGetIntervalMatrix(fuzzyMatrix5, {intervalMatrix5a, {{0, 1}, {0, 2}, {2, 0}, {2, 3}, {2, 4}, {4, 0}, {4, 2}, {4, 3}}}, 0.82) << std::endl;
    std::cout << testGetIntervalMatrix(fuzzyMatrix5, {intervalMatrix5b, {{0, 1}, {0, 2}, {2, 0}, {2, 3}, {2, 4}, {4, 0}, {4, 2}, {4, 3}}}, 0.53) << std::endl;

    // getAlphaCut
    std::cout << testGetAlphaCut(storm::storage::TriangularFuzzyNumber(0.21, 0.37, 0.42), storm::Interval(0.21, 0.42), 0) << std::endl;
    std::cout << testGetAlphaCut(storm::storage::TriangularFuzzyNumber(0.21, 0.37, 0.42), storm::Interval(0.29, 0.395), 0.5) << std::endl;
    std::cout << testGetAlphaCut(storm::storage::TriangularFuzzyNumber(0.21, 0.37, 0.42), storm::Interval(0.37), 1) << std::endl;

    // getFeasibleMutationRange
    std::cout << testGetFeasibleMutationRange(0.21, 0.42, storm::Interval(0.13, 0.37), storm::Interval(0.37, 0.69), storm::Interval(-0.08, 0.05)) << std::endl;
    std::cout << testGetFeasibleMutationRange(0.66, 0.33, storm::Interval(0.66, 1.0), storm::Interval(0.0, 0.33), storm::Interval(0.0, 0.33)) << std::endl;
    std::cout << testGetFeasibleMutationRange(0.0, 1.0, storm::Interval(0.0, 1.0), storm::Interval(0.0, 1.0), storm::Interval(0.0, 1.0)) << std::endl;
    std::cout << testGetFeasibleMutationRange(0.5, 0.5, storm::Interval(0.0, 1.0), storm::Interval(0.0, 1.0), storm::Interval(-0.5, 0.5)) << std::endl;
    std::cout << testGetFeasibleMutationRange(0.5, 0.5, storm::Interval(0.5, 0.5), storm::Interval(0.0, 1.0), storm::Interval(0.0, 0.0)) << std::endl;
    std::cout << testGetFeasibleMutationRange(0.5, 0.5, storm::Interval(0.5, 0.5), storm::Interval(0.5, 0.5), storm::Interval(0.0, 0.0)) << std::endl;

    /*
    storm::storage::FuzzyAnalysisResult::member member3(crispMatrix3, 0, {0, 0});
    print2DVector(mutateMember(member3, {{0, 1}, {0, 2}, {2, 0}, {2, 2}}, {0, 2}, intervalMatrix3).getMatrix());
    std::cout << std::endl;
    storm::storage::FuzzyAnalysisResult::member member4(crispMatrix4, 0, {0, 0});
    print2DVector(mutateMember(member4, {{1, 0}, {1, 2}, {1, 3}, {3, 0}, {3, 1}}, {1, 3}, intervalMatrix4).getMatrix());
    std::cout << std::endl;
    storm::storage::FuzzyAnalysisResult::member member5a(crispMatrix5, 0, {0, 0});
    print2DVector(mutateMember(member5a, {{0, 1}, {0, 2}, {2, 0}, {2, 3}, {2, 4}, {4, 0}, {4, 2}, {4, 3}}, {0, 2, 4}, intervalMatrix5a).getMatrix());
    std::cout << std::endl;
    storm::storage::FuzzyAnalysisResult::member member5b(crispMatrix5, 0, {0, 0});
    print2DVector(mutateMember(member5b, {{0, 1}, {0, 2}, {2, 0}, {2, 3}, {2, 4}, {4, 0}, {4, 2}, {4, 3}}, {0, 2, 4}, intervalMatrix5b).getMatrix());
    std::cout << std::endl;
    */

    /*
    storm::storage::FuzzyAnalysisResult test(fuzzyMatrix5);
    storm::storage::FuzzyAnalysisResult::member member5c(crispMatrix5, 0, {0, 0});
    storm::storage::FuzzyAnalysisResult::member member5d(crispMatrix5alt, 0, {0, 0});
    std::vector<storm::storage::FuzzyAnalysisResult::member> children = test.crossParents(member5c, member5d, {0, 2, 4});

    std::cout << "PARENT 1" << std::endl;
    print2DVector(member5c.getMatrix());
    std::cout << std::endl;
    std::cout << "PARENT 2" << std::endl;
    print2DVector(member5d.getMatrix());
    std::cout << std::endl;
    std::cout << "CHILD 1" << std::endl;
    print2DVector(children[0].getMatrix());
    std::cout << std::endl;
    std::cout << "CHILD 2" << std::endl;
    print2DVector(children[1].getMatrix());
    std::cout << std::endl;
    */

    std::vector<std::vector<double>> regular_test = {{0.5, 0.0, 0.5}, {0.25, 0.75, 0.0}, {0.0, 0.6, 0.4}};
    std::vector<std::vector<double>> irregular_test1 = {{0.5, 0.0, 0.5}, {0.25, 0.0, 0.0}, {0.0, 0.0, 0.4}};
    std::vector<std::vector<double>> irregular_test2 = {{0.0, 0.5, 0.0}, {0.25, 0.0, 0.0}, {0.0, 0.0, 0.4}};
    std::vector<std::vector<double>> irregular_test3 = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    std::vector<std::vector<double>> absorbing_test1 = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    std::vector<std::vector<double>> absorbing_test2 = {{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.5, 0.0}};
    std::vector<std::vector<double>> absorbing_test3 = {{1.0, 0.0, 0.0}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
    std::vector<std::vector<double>> non_absorbing_test = {{1.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.5, 0.0}};

    std::cout << is_regular(regular_test) << std::endl;
    std::cout << !is_regular(irregular_test1) << std::endl;
    std::cout << !is_regular(irregular_test2) << std::endl;
    std::cout << !is_regular(irregular_test3) << std::endl;

    std::cout << !is_absorbing(regular_test) << std::endl;
    std::cout << is_absorbing(absorbing_test1) << std::endl;
    std::cout << is_absorbing(absorbing_test2) << std::endl;
    std::cout << is_absorbing(absorbing_test3) << std::endl;
    std::cout << !is_absorbing(non_absorbing_test) << std::endl;

    /*
    std::vector<double> pi = stationary_distribution(regular_test);
    for (int i = 0; i < pi.size(); i++)
    {
        std::cout << pi[i] << " ";
    }
    std::cout << std::endl;
    */
}