#include <storm/api/storm.h>
#include <storm/utility/initialize.h>
#include <storm/models/sparse/Dtmc.h>
#include <storm/storage/TriangularFuzzyNumber.h>
#include <storm/storage/FuzzyAnalysisResult.h>
#include <vector>

typedef storm::models::sparse::Dtmc<storm::storage::TriangularFuzzyNumber> Dtmc;
typedef storm::modelchecker::SparseDtmcPrctlModelChecker<Dtmc> DtmcModelChecker;

template<typename ValueType>
storm::storage::SparseMatrix<ValueType> createMatrix() {
    storm::storage::SparseMatrixBuilder<ValueType> matrixBuilder;
    ValueType prob = storm::storage::TriangularFuzzyNumber(0.4,0.5,0.6);
    matrixBuilder.addNextValue(0, 1, prob);
    matrixBuilder.addNextValue(0, 2, storm::storage::TriangularFuzzyNumber(0.4,0.5,0.6));
    matrixBuilder.addNextValue(1, 0, storm::storage::TriangularFuzzyNumber(0.1,0.2,0.3));
    matrixBuilder.addNextValue(1, 3, storm::storage::TriangularFuzzyNumber(0.7,0.8,0.9));
    matrixBuilder.addNextValue(2, 2, storm::storage::TriangularFuzzyNumber(0.1,0.2,0.3));
    matrixBuilder.addNextValue(2, 4, storm::storage::TriangularFuzzyNumber(0.7,0.8,0.9));
    matrixBuilder.addNextValue(3, 3, storm::storage::TriangularFuzzyNumber(0,1.0,0));
    matrixBuilder.addNextValue(4, 4, storm::storage::TriangularFuzzyNumber(0,1.0,0));

    return matrixBuilder.build();
}

std::vector<std::vector<double>> createMatrix2() {
    std::vector<std::vector<double>> v;
    v.push_back({0, 0.5, 0.5, 0, 0});
    v.push_back({0.2, 0, 0, 0.8, 0});
    v.push_back({0, 0, 0.2, 0, 0.8});
    v.push_back({0, 0, 0, 1, 0});
    v.push_back({0, 0, 0, 0, 1});

    return v;
}

template<typename ValueType, typename RewardModelType>
std::shared_ptr<storm::models::sparse::Model<ValueType, RewardModelType>> createModel(storm::storage::SparseMatrix<ValueType> matrix, size_t goalState) {
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

void check() {
    // Create Matrix
    auto matrix = createMatrix<storm::storage::TriangularFuzzyNumber>();
    auto matrix2 = createMatrix2();
    
    std::cout << matrix << std::endl;
    storm::storage::FuzzyAnalysisResult test(matrix);
    std:: cout << "Alpha cut: " << test.getAlphaCut(storm::storage::TriangularFuzzyNumber(0.4,0.5,0.6), 0.5) << std::endl;
    test.isFeasible(matrix2, 0.5);
    
    // Set goal state
    size_t goalState = 4;
    // Create model
    auto model = createModel<storm::storage::TriangularFuzzyNumber, storm::models::sparse::StandardRewardModel<storm::storage::TriangularFuzzyNumber>>(matrix, goalState);

    // Print information about model
    model->printModelInformationToStream(std::cout);
    matrix = model->getTransitionMatrix();
    std::cout << matrix << std::endl;
}


int main (int argc, char *argv[]) {
    if (argc > 1) {
        std::cout << "Arguments are ignored" << std::endl;
    }

    // Init loggers
    storm::utility::setUp();
    // Set some settings objects.
    storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

    // Call function
    check();
}
