#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric>

#include <storm/storage/TriangularFuzzyNumber.h>
#include <storm/adapters/RationalFunctionAdapter.h>

bool is_regular(std::vector<std::vector<double>> P)
{
    int n = P.size();

    std::vector<int> reachable(n);
    reachable[0] = 1;
    for (int k = 0; k < n; k++)
    {
        std::vector<int> new_reachable(reachable);
        for (int i = 0; i < n; i++)
        {
            if (reachable[i] == 1)
            {
                for (int j = 0; j < n; j++)
                {
                    if (P[i][j] > 0)
                    {
                        new_reachable[j] = 1;
                    }
                }
            }
        }
        reachable = new_reachable;
    }
    bool irreducible = true;
    for (int i = 0; i < n; i++)
    {
        if (reachable[i] == 0)
        {
            irreducible = false;
            break;
        }
    }
    if (!irreducible)
        return false;

    std::vector<int> d(n);
    for (int i = 0; i < n; i++)
    {
        d[i] = 1;
        for (int k = 1;; k++)
        {
            if (P[i][i] > 0)
            {
                d[i] = k;
                break;
            }
            std::vector<double> p(n);
            for (int j = 0; j < n; j++)
            {
                for (int l = 0; l < n; l++)
                {
                    p[j] += P[l][j] * P[i][l];
                }
            }
            P[i] = p;
        }
    }
    bool aperiodic = true;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (std::__gcd(d[i], d[j]) != 1)
            {
                aperiodic = false;
                break;
            }
        }
        if (!aperiodic)
            break;
    }
    if (!aperiodic)
        return false;

    return true;
}

bool is_absorbing(std::vector<std::vector<double>> P)
{
    int n = P.size();
    int k = 0;
    std::vector<int> absorbing_states;
    for (int i = 0; i < n; i++)
    {
        if (P[i][i] == 1)
        {
            absorbing_states.push_back(i);
            k++;
        }
    }
    if (k == 0)
    {
        return false;
    }

    for (int i = 0; i < n; i++)
    {
        bool reachable = false;
        for (int j = 0; j < absorbing_states.size(); j++)
        {
            if (i == absorbing_states[j])
            {
                reachable = true;
                break;
            }
        }
        if (!reachable)
        {
            std::vector<double> p(n);
            for (int k = 0; k < n; k++)
            {
                p[k] = P[i][k];
            }
            for (int k = 0; k < n; k++)
            {
                for (int l = 0; l < n; l++)
                {
                    p[k] += P[i][l] * P[l][k];
                }
            }
            for (int j = 0; j < absorbing_states.size(); j++)
            {
                if (p[absorbing_states[j]] > 0)
                {
                    reachable = true;
                    break;
                }
            }
            if (!reachable)
            {
                return false;
            }
        }
    }

    return true;
}

std::vector<double> stationary_distribution(std::vector<std::vector<double>> P)
{
    int n = P.size();
    std::vector<double> w(n);
    for (int i = 0; i < n; i++)
    {
        w[i] = 1.0;
    }

    std::vector<double> prev_w(n);
    double prev_diff = 0;
    double diff = 1;
    while (diff > 1e-8)
    {
        prev_w = w;
        for (int i = 0; i < n; i++)
        {
            w[i] = 0;
            for (int j = 0; j < n; j++)
            {
                w[i] += prev_w[j] * P[j][i];
            }
        }

        prev_diff = diff;
        diff = 0;
        for (int i = 0; i < n; i++)
        {
            diff += abs(w[i] - prev_w[i]);
        }
    }

    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += w[i];
    }
    for (int i = 0; i < n; i++)
    {
        w[i] /= sum;
    }

    return w;
}

storm::Interval getAlphaCut(storm::storage::TriangularFuzzyNumber const &t, double alpha)
{
    double i = alpha * (t.getPeak() - t.getLeftBound()) + t.getLeftBound();
    double j = (1 - alpha) * (t.getRightBound() - t.getPeak()) + t.getPeak();
    return storm::Interval(i, j);
}

double matrixMul(std::vector<std::vector<double>> matrix, int p, int rX, int rY)
{
    const int n = matrix.size();
    const int m = matrix[0].size();
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0));
    std::vector<std::vector<double>> temp = matrix;

    for (auto l = 0; l < p - 1; ++l)
    {
        for (auto j = 0; j < m; ++j)
        {
            for (auto k = 0; k < m; ++k)
            {
                for (auto i = 0; i < n; ++i)
                {
                    result[i][j] += temp[i][k] * matrix[k][j];
                }
            }
        }
        temp = result;
        for (auto &i : result)
            std::fill(i.begin(), i.end(), 0);
    }

    return temp[rX][rY];
}

std::vector<std::vector<storm::Interval>> getIntervalMatrix(storm::storage::FlexibleSparseMatrix<storm::storage::TriangularFuzzyNumber> matrix, double alpha)
{
    std::vector<std::vector<storm::Interval>> intervalMatrix(matrix.getRowCount(), std::vector<storm::Interval>(matrix.getColumnCount(), storm::Interval(0)));

    for (storm::storage::FlexibleSparseMatrix<storm::storage::TriangularFuzzyNumber>::index_type row = 0; row < matrix.getRowCount(); ++row)
    {
        for (auto const element : matrix.getRow(row))
        {
            int column = element.getColumn();
            storm::storage::TriangularFuzzyNumber value = element.getValue();
            if (value.getLeftBound() == 0 && value.getRightBound() == 0)
            { // crisp number
                intervalMatrix[row][column] = storm::Interval(value.getPeak());
            }
            else
            {
                intervalMatrix[row][column] = getAlphaCut(value, alpha);
            }
        }
    }

    return intervalMatrix;
}

bool is_valid_matrix(const std::vector<std::vector<double>> &matrix, double tolerance)
{
    for (const auto &row : matrix)
    {
        if (std::abs(std::accumulate(row.begin(), row.end(), 0.0) - 1.0) >= tolerance)
        {
            return false;
        }
    }
    return true;
}

void printMatrix(std::vector<std::vector<double>> matrix)
{
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[i].size(); j++)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> bruteForceMatrixMul(const std::vector<std::vector<storm::Interval>> &intervalMatrix, std::vector<std::vector<double>> &current_matrix, double &best_result, int i, int j, int steps, std::pair<int, int> idx, bool direction, double tolerance = 0.005, double iterStep = 0.005)
{
    if (i == intervalMatrix.size())
    {
        if (is_valid_matrix(current_matrix, tolerance))
        {
            double current_result = matrixMul(current_matrix, steps, idx.first, idx.second);
            if ((direction && current_result < best_result) || (!direction && current_result > best_result))
            {
                best_result = current_result;
                std::cout << best_result << std::endl;
                printMatrix(current_matrix);
                return current_matrix;
            }
        }
        return {};
    }

    std::vector<std::vector<double>> best_matrix;
    for (double x = intervalMatrix[i][j].lower(); x <= intervalMatrix[i][j].upper(); x += iterStep)
    {
        current_matrix[i][j] = x;
        auto new_best_matrix = bruteForceMatrixMul(intervalMatrix, current_matrix, best_result, i + (j + 1) / intervalMatrix[i].size(), (j + 1) % intervalMatrix[i].size(), steps, idx, direction, tolerance, iterStep);
        if (!new_best_matrix.empty())
        {
            best_matrix = new_best_matrix;
        }
    }
    return best_matrix;
}

void reachableInSteps(storm::storage::FlexibleSparseMatrix<storm::storage::TriangularFuzzyNumber> matrix, std::pair<int, int> idx, int steps, int acc, double tolerance = 0.005, double iterStep = 0.005)
{
    for (int i = 0; i < acc; i++)
    {
        double alpha = (double)i / (double)acc;
        std::cout << "alpha=" << alpha << " processing... " << std::endl;
        std::vector<std::vector<storm::Interval>> intervalMatrix = getIntervalMatrix(matrix, alpha);
        std::vector<std::vector<double>> emptyMatrix(intervalMatrix.size(), std::vector<double>(intervalMatrix[0].size()));
        double one = 1.0;
        double zero = 0.0;
        std::vector<std::vector<double>> result1 = bruteForceMatrixMul(intervalMatrix, emptyMatrix, one, 0, 0, steps, idx, true, tolerance, iterStep);
        std::vector<std::vector<double>> result2 = bruteForceMatrixMul(intervalMatrix, emptyMatrix, zero, 0, 0, steps, idx, false, tolerance, iterStep);
        printMatrix(result1);
        printMatrix(result2);
        std::cout << "alpha=" << alpha << ", left: " << matrixMul(result1, steps, idx.first, idx.second) << ", right: " << matrixMul(result2, steps, idx.first, idx.second) << std::endl;
    }
}
