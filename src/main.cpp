#include <storm/api/storm.h>
#include <storm/utility/initialize.h>
#include <storm/models/sparse/Dtmc.h>
#include <storm/storage/TriangularFuzzyNumber.h>
#include <storm/storage/FuzzyAnalysisResult.h>
#include <storm/adapters/RationalFunctionAdapter.h>
#include <vector>
#include "unit_tests.cpp"

typedef storm::models::sparse::Dtmc<storm::storage::TriangularFuzzyNumber> Dtmc;
typedef storm::modelchecker::SparseDtmcPrctlModelChecker<Dtmc> DtmcModelChecker;

storm::storage::SparseMatrix<storm::storage::TriangularFuzzyNumber> createMatrix()
{
    storm::storage::SparseMatrixBuilder<storm::storage::TriangularFuzzyNumber> matrixBuilder;
    matrixBuilder.addNextValue(0, 0, storm::storage::TriangularFuzzyNumber(0.6, 0.7, 0.8));
    matrixBuilder.addNextValue(0, 1, storm::storage::TriangularFuzzyNumber(0.2, 0.3, 0.4));
    matrixBuilder.addNextValue(1, 0, storm::storage::TriangularFuzzyNumber(0.3, 0.4, 0.5));
    matrixBuilder.addNextValue(1, 1, storm::storage::TriangularFuzzyNumber(0.5, 0.6, 0.7));

    return matrixBuilder.build();
}

std::vector<std::vector<double>> createMatrix2()
{
    std::vector<std::vector<double>> v;
    v.push_back({0, 0.5, 0.5, 0, 0});
    v.push_back({0.2, 0, 0, 0.8, 0});
    v.push_back({0, 0, 0.2, 0, 0.8});
    v.push_back({0, 0, 0, 1, 0});
    v.push_back({0, 0, 0, 0, 1});

    return v;
}

template <typename ValueType, typename RewardModelType>
std::shared_ptr<storm::models::sparse::Model<ValueType, RewardModelType>> createModel(storm::storage::SparseMatrix<ValueType> matrix, size_t goalState)
{
    // Initialize
    size_t stateSize = matrix.getColumnCount();
    auto modelComponents = std::make_shared<storm::storage::sparse::ModelComponents<ValueType, RewardModelType>>();
    modelComponents->stateLabeling = storm::models::sparse::StateLabeling(stateSize);

    // Set labels
    // First state is initial state
    modelComponents->stateLabeling.addLabel("init");
    modelComponents->stateLabeling.addLabelToState("init", 0);
    // Set goal state
    modelComponents->stateLabeling.addLabel("goal");
    modelComponents->stateLabeling.addLabelToState("goal", goalState);

    // Build transition matrix
    modelComponents->transitionMatrix = matrix;

    // Build model
    return storm::utility::builder::buildModelFromComponents(storm::models::ModelType::Dtmc, std::move(*modelComponents));
}

void check()
{
    // Create Matrix
    auto matrix = createFuzzyMatrix4();
    std::cout << matrix << std::endl;
    storm::storage::FuzzyAnalysisResult test(matrix);
    test.fuzzyGeneticAlgorithm({1, 3}, 5, 4, 200, 1000, 0.2, 0.2, false);

    /*
    // Set goal state
    size_t goalState = 4;
    // Create model
    auto model = createModel<storm::storage::TriangularFuzzyNumber, storm::models::sparse::StandardRewardModel<storm::storage::TriangularFuzzyNumber>>(matrix, goalState);

    // Print information about model
    model->printModelInformationToStream(std::cout);
    matrix = model->getTransitionMatrix();
    std::cout << matrix << std::endl;
    */
}

void printTFNValues(std::vector<std::pair<double, double>> tfn) {
    std::cout << "x\t\t y" << std::endl;
    for (const auto &entry : tfn)
    {
        std::cout << entry.first << " " << entry.second << std::endl;
    }
    std::cout << std::endl;
}

void demonstrate() {
    auto randomFuzzyMatrix = generateRandomFuzzyMatrix(10);
    std::cout << randomFuzzyMatrix << std::endl;

    storm::storage::FuzzyAnalysisResult analysisResult(randomFuzzyMatrix);
    //std::vector<std::pair<double, double>> reachability = analysisResult.fuzzyGeneticAlgorithm({0, 1}, 5, 10, 50, 25000, 0.15, 0.2, true);

    bool regular = analysisResult.isRegular();
    bool absorbing = analysisResult.isAbsorbing();
    std::cout << "Regular? " << regular << ", Absorbing? " << absorbing << std::endl;

    if(regular){
        //std::vector<std::pair<double, double>> stationary = analysisResult.fuzzyGeneticAlgorithm({0, 0}, 5, 2, 50, 25000, 0.15, 0.2, true, false);
        //printTFNValues(stationary);
    }

    //printTFNValues(reachability);
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        std::cout << "Arguments are ignored" << std::endl;
    }

    // Init loggers
    storm::utility::setUp();
    // Set some settings objects.
    storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

    // Call function
    // check();
    // run_benchmarks();
    demonstrate();
}
