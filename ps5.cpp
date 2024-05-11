#include <iostream>
#include <bitset>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Eigenvalues>
#include <bitset>
#include "lapacke.h"

typedef Eigen::VectorXd vec;
typedef Eigen::MatrixXd mat;

std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_eigensystem(const Eigen::MatrixXd &K)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigenvalue decomposition failed.");

    return {solver.eigenvalues(), solver.eigenvectors()};
}
// template <typename T>
// std::pair<std::vector<T>, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solve_sparse_eigensystem(const Eigen::SparseMatrix<T> &A)
// {
//     // Check if the matrix size is consistent
//     int n = static_cast<int>(A.rows()); // Assumes square matrix

//     // Declare solver from Eigen library, for symmetric matrices, use Eigen::SimplicialLDLT or Eigen::SparseLU for non-symmetric
//     Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;
//     solver.compute(A);

//     if (solver.info() != Eigen::Success)
//     {
//         std::cerr << "Decomposition failed." << std::endl;
//         throw std::runtime_error("Matrix decomposition failed.");
//     }

//     // Assuming we are only interested in solving Ax = lambda*x (i.e., solving for eigenvalues directly is not directly supported like dsyev)
//     // Eigenvalues need to be found indirectly if necessary, typically through iterative methods or by transforming the problem

//     // Example of solving for a specific eigenvalue (smallest magnitude eigenvalue here as an example)
//     Eigen::Matrix<T, Eigen::Dynamic, 1> x = Eigen::Matrix<T, Eigen::Dynamic, 1>::Random(n);
//     T lambda = x.transpose() * A * x / x.transpose() * x; // Rayleigh quotient for an approximation

//     // Here we assume that eigenvectors are not computed directly, for demonstration:
//     std::vector<T> eigenvalues{lambda}; // Placeholder for demonstration

//     // Eigen does not compute eigenvectors directly in the Sparse solver as done in LAPACKE_dsyev
//     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigenvectors(n, 1); // Placeholder for a single eigenvector
//     eigenvectors.col(0) = x.normalized();

//     return {eigenvalues, eigenvectors};
// }
// Modify the Davidson class to handle multiple eigenvectors
inline size_t assert_equivalence(size_t s1, size_t s2, const char *f = "unspecified function")
{
    if (s1 != s2)
    {
        std::stringstream ss;
        ss << f << ": " << s1 << " " << s2 << " should be equal.";
        throw ss.str();
    }
    else
        return s1;
}

inline size_t assert_equivalence(size_t s1, size_t s2, size_t s3, const char *f = "unspecified function")
{
    if (s1 != s2 || s1 != s3)
    {
        std::stringstream ss;
        ss << f << ": " << s1 << " " << s2 << " " << s3 << " should be equal.";
        throw ss.str();
    }
    return s1;
}

// class Davidson
// {
// protected:
//     std::vector<vec> basis, Hbasis; // vec is like valarray <double>; use the vector class of your choosing
//     std::vector<double> saved_energies;
//     mutable mat K;

// public:
//     size_t basis_size() const { return basis.size(); }
//     size_t full_dim() const { return basis[0].size(); }
//     double get_energy() const { return saved_energies.back(); }
//     static bool check_tol(double a, double b) { return std::abs(a - b) < 1e-9; }
//     bool has_stopped() const
//     {
//         // Do a double-check on the energies for extra safety
//         const auto &x = saved_energies;
//         const int n = x.size();
//         return n >= 3 && check_tol(x[n - 1], x[n - 2]) && check_tol(x[n - 1], x[n - 3]);
//     }

//     explicit Davidson(size_t n)
//     {
//         // Initialize basis with normalized random vector
//         vec x = gen_ran_vec(n);
//         basis.emplace_back(x /= sqrt(x.dot(x)));
//     }

//     const mat &update_K() const // mat is a simple dense matrix class; use one of your choosing
//     {
//         const int
//             nold = assert_equivalence(K.rows(), K.cols(), "K must be a square matrix"),
//             n = assert_equivalence(basis.size(), Hbasis.size(), "update_K");

//         // Incremental update to subspace matrix
//         mat Kold = K;
//         K.resize(n, n);

//         // Copy old matrix elements
//         for (int i = 0; i < nold; ++i)
//             for (int j = 0; j < nold; ++j)
//                 K(i, j) = Kold(i, j);

//         // Calculate (and save) new row and column
//         for (int j = nold; j < n; ++j)
//             for (int i = 0; i < j + 1; ++i)
//                 K(j, i) = K(i, j) = basis[i].dot( Hbasis[j]);

//         return K;
//     }

//     template <typename op> // op is your sparse matrix which provides  a matrix-vector "dot" product
//     void do_iter(const op &H)
//     {
//         // Check sizes: your op needs to provide the number of rows and cols it represents
//         assert_equivalence(full_dim(), H.rows(), H.cols(), "do_iter (H)");

//         // "dot" (ie multiply) H (Note that H is only applied ONCE per iteration and saved)
//         Hbasis.emplace_back(H.dot(basis.back())); // op must implement "dot"

//         // Solve subspace problem (after updating K)
//         const auto [E, U] = sjc::lapack::solve_eigensystem(update_K()); // You have this from PS4
//         saved_energies.push_back(E[0]);

//         // Construct the diagonally-preconditioned residual vector
//         const vec c = U.get_col(0);       // Your dense matrix must provide a column-vector getter
//         const auto &D = H.get_diagonal(); // op must provide the diagonal D
//         vec x(0, full_dim());             // Note arguments to constructor reversed versus std::vector
//         for (int j = 0; j < c.size(); ++j)
//         {
//             const vec &Hb = Hbasis[j], &b = basis[j];

//             for (int k = 0; k < full_dim(); ++k)
//                 x[k] += c[j] * (Hb[k] - E[0] * b[k]) / (E[0] - D[k]);
//         }

//         // Orthogonalize: you need to define an inner product operation on two vecs
//         for (int j = 0; j < basis.size(); ++j)
//             x -= inner_product(x, basis[j]) * basis[j];

//         // Normalize and save (again, uses your inner_product)
//         basis.push_back(x /= sqrt(inner_product(x, x)));
//     }

//     template <typename op>
//     Davidson &build_for_ground_state(const op &H)
//     {
//         for (int iter = 0; iter < full_dim() && !has_stopped(); ++iter)
//             do_iter(H);
//         return *this;
//     }
// };

class Davidson
{
protected:
    std::vector<vec> basis, Hbasis;
    std::vector<double> saved_energies;
    mutable mat K;

public:
    Davidson(size_t n)
    {
        // Initialize basis with one normalized random vector
        vec x = vec::Random(n);
        x.normalize(); // Normalize the vector
        basis.push_back(x);
    }

    size_t basis_size() const { return basis.size(); }
    size_t full_dim() const { return basis[0].size(); }
    double get_energy() const { return saved_energies.back(); }
    static bool check_tol(double a, double b) { return std::abs(a - b) < 1e-9; }

    // Other methods need to be adapted similarly...

    bool has_stopped() const
    {
        if (saved_energies.size() < 3)
            return false;
        double last = saved_energies.back();
        return std::abs(last - saved_energies[saved_energies.size() - 2]) < 1e-10 &&
               std::abs(last - saved_energies[saved_energies.size() - 3]) < 1e-10;
    }

    const mat &update_K() const
    {
        // Example adaptation for updating K
        int n = basis.size();
        K = mat(n, n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                K(i, j) = K(j, i) = basis[i].dot(Hbasis[j]);
            }
        }
        return K;
    }
    template <typename op>
    void do_iter(const op &H)
    {
        assert_equivalence(full_dim(), H.rows(), H.cols(), "do_iter (H)");
        Hbasis.emplace_back(H * basis.back());

        auto [E, U] = solve_eigensystem(update_K());
        saved_energies.push_back(E[0]);

        const vec &c = U.col(0);
        vec D = H.diagonal();
        vec x = vec::Zero(full_dim());
        for (int j = 0; j < c.size(); ++j)
        {
            const vec &Hb = Hbasis[j];
            const vec &b = basis[j];
            x += c(j) * (Hb - E[0] * b).cwiseQuotient(vec::Constant(full_dim(), E[0]) - D);
        }

        for (int j = 0; j < basis.size(); ++j)
            x -= basis[j] * x.dot(basis[j]);
        x.normalize();
        basis.push_back(x);
    }

    template <typename op>
    Davidson &build_for_ground_state(const op &H)
    {
        for (int iter = 0; iter < full_dim() && !has_stopped(); ++iter)
        {
            do_iter(H);
        }
        return *this;
    }
};

class DavidsonSEM : public Davidson
{
public:
    size_t target_eigenpairs;      // Number of eigenpairs to solve for
    std::vector<vec> eigenvectors; // Vector to store eigenvectors

    DavidsonSEM(size_t n) : Davidson(n), target_eigenpairs(n)
    {
        eigenvectors.resize(n);
        saved_energies.resize(n);
    }

    template <typename op>
    void do_iter(const op &H)
    {
        auto [E, U] = solve_eigensystem(update_K()); // Diagonalize subspace matrix
        for (size_t idx = 0; idx < target_eigenpairs; ++idx)
        {
            vec D = H.diagonal();
            saved_energies[idx] = E[idx];
            vec c = U.col(idx);
            vec x = vec::Zero(full_dim());

            for (int j = 0; j < c.size(); ++j)
            {
                const vec &Hb = Hbasis[j], &b = basis[j];
                for (int k = 0; k < full_dim(); ++k)
                    x[k] += c[j] * (Hb[k] - E[idx] * b[k]) / (E[idx] - D[k]);
            }

            // Orthogonalization and normalization steps
            x = orthogonalize(x, basis);
            x.normalize();
            eigenvectors[idx] = x;
        }
    }

    void build_for_eigenstates(const Eigen::SparseMatrix<double> &H, int num_states)
    {
        target_eigenpairs = num_states;
        eigenvectors.resize(num_states);
        saved_energies.resize(num_states);
        while (!has_stopped())
        {
            do_iter(H);
        }
    }

private:
    vec orthogonalize(const vec &v, const std::vector<vec> &basis)
    {
        vec ortho_v = v;
        for (const auto &b : basis)
        {
            ortho_v -= b * v.dot(b);
        }
        return ortho_v;
    }
};
// Extend build_for_ground_state to handle multiple eigenpairs

class ElectronicConfiguration
{
private:
    std::bitset<64> config;

public:
    // Constructor that initializes the configuration with N2 molecule's ground state
    ElectronicConfiguration()
    {
        // Set spin-up electrons for σ orbitals (lower energy to higher, indices 0 to 6)
        for (int i = 0; i < 7; ++i)
        {
            config.set(i); // Spin-up electrons in σ orbitals
        }

        // Set spin-down electrons for σ orbitals (next 18 bits, indices 18 to 24)
        for (int i = 18; i < 25; ++i)
        {
            config.set(i); // Spin-down electrons in σ orbitals
        }
    }

    // Function to flip a single bit (for creating single excitations)
    void flipSingle(int position)
    {
        config.flip(position);
    }

    // Function to flip two bits (for creating double excitations)
    void flipDouble(int position1, int position2)
    {
        config.flip(position1);
        config.flip(position2);
    }

    // Get the current configuration as a string (for output and debugging)
    std::string getConfigAsString() const
    {
        return config.to_string();
    }

    // Generate single and double excitations
    std::vector<std::bitset<64>> generateExcitations()
    {
        std::vector<std::bitset<64>> excitations;
        // Generate all single excitations
        for (size_t i = 0; i < 64; ++i)
        {
            if (config.test(i))
            {
                std::bitset<64> newConfig = config;
                newConfig.flip(i);
                excitations.push_back(newConfig);

                // Generate all double excitations from each single excitation
                for (size_t j = i + 1; j < 64; ++j)
                {
                    if (config.test(j))
                    {
                        std::bitset<64> doubleConfig = newConfig;
                        doubleConfig.flip(j);
                        excitations.push_back(doubleConfig);
                    }
                }
            }
        }
        return excitations;
    }
};

std::vector<std::bitset<64>> generateConfigurations(const std::bitset<64> &reference)
{
    std::vector<std::bitset<64>> configurations;
    configurations.push_back(reference); // Add the reference configuration

    // Generate all single and double excitations
    for (size_t i = 0; i < 64; ++i)
    {
        if (reference.test(i)) // Check if orbital i is occupied
        {
            std::bitset<64> singleExcitation = reference;
            singleExcitation.flip(i); // Generate single excitation
            configurations.push_back(singleExcitation);

            // Generate double excitations
            for (size_t j = i + 1; j < 64; ++j)
            {
                if (reference.test(j))
                {
                    std::bitset<64> doubleExcitation = singleExcitation;
                    doubleExcitation.flip(j);
                    configurations.push_back(doubleExcitation);
                }
            }
        }
    }
    return configurations;
}

typedef Eigen::SparseMatrix<double> SpMat;

SpMat createSparseMatrix(int size)
{
    SpMat mat(size, size);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(size * 3);

    for (int i = 0; i < size; ++i)
    {
        tripletList.emplace_back(i, i, i + 1.0); // Diagonal elements
        if (i + 1 < size)
        {
            tripletList.emplace_back(i, i + 1, -1.0); // Upper diagonal
            tripletList.emplace_back(i + 1, i, -1.0); // Lower diagonal
        }
    }

    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}
double computeMatrixElement(const std::bitset<64> &conf1, const std::bitset<64> &conf2, const Integrals &integrals)
{
    // Pseudo code: determine the type of excitation and call the appropriate integral function
    if (conf1 == conf2)
    {
        return integrals.diagonal_element(conf1); // Diagonal elements
    }
    else
    {
        int diff = (conf1 ^ conf2).count(); // Count the number of different bits
        if (diff == 2)
        {
            return integrals.single_excitation(conf1, conf2);
        }
        else if (diff == 4)
        {
            return integrals.double_excitation(conf1, conf2);
        }
    }
    return 0;
}

SpMat buildHamiltonianMatrix(const vector<bitset<64>> &configurations, const Integrals &integrals)
{
    SpMat hamiltonian(configurations.size(), configurations.size());

    for (size_t i = 0; i < configurations.size(); ++i)
    {
        for (size_t j = i; j < configurations.size(); ++j)
        {
            double value = computeMatrixElement(configurations[i], configurations[j], integrals);
            if (value != 0)
            {
                hamiltonian.insert(i, j) = value;
                if (i != j)
                {
                    hamiltonian.insert(j, i) = value; // Hamiltonian is Hermitian
                }
            }
        }
    }
    return hamiltonian;
}

std::vector<std::bitset<64>> generateConfigurations(const ElectronicConfiguration &referenceConfig)
{
    std::vector<std::bitset<64>> configurations;

    // add ref
    std::bitset<64> referenceBitset(referenceConfig.getConfigAsString(), 0, 64, '0', '1');
    configurations.push_back(referenceBitset);

    // gen songle and sdouble
    for (size_t i = 0; i < 64; ++i)
    {
        if (referenceBitset.test(i))
        { // i?
            std::bitset<64> singleExcitation = referenceBitset;
            singleExcitation.flip(i);
            configurations.push_back(singleExcitation);

            for (size_t j = i + 1; j < 64; ++j)
            {
                if (referenceBitset.test(j))
                {
                    std::bitset<64> doubleExcitation = singleExcitation;
                    doubleExcitation.flip(j);
                    configurations.push_back(doubleExcitation);
                }
            }
        }
    }
    return configurations;
}

int main()
{
    // Example range of bond lengths
    std::vector<double> bond_lengths = {0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6};
    std::vector<double> energies;

    for (double R : bond_lengths)
    {
        Eigen::SparseMatrix<double> H = createHamiltonianMatrix(R); // Define this function based on your Hamiltonian generation
        DavidsonSEM dav(64);                                        // Assuming a 64 orbital system
        dav.build_for_multiple_states(H, 6);                        // Get ground and 5 excited states

        double core_energy = get_core_energy(R);
        for (int i = 0; i < 6; ++i)
        {
            double total_energy = dav.get_energy(i) + core_energy; // Assuming `get_energy(i)` gets the i-th energy
            energies.push_back(total_energy);
        }

        // Output or store energies for plotting
    }
    td::vector<double> bond_lengths = {0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6};
    std::vector<double> energies;

    for (double R : bond_lengths)
    {
        Eigen::SparseMatrix<double> H = createHamiltonianMatrix(R); // Implementation needed
        DavidsonSEM dav(64);
        dav.build_for_multiple_states(H, 6);

        double core_energy = get_core_energy(R); // Implementation needed
        std::cout << "Bond length " << R << " Angstroms:\n";
        for (int i = 0; i < 6; ++i)
        {
            double total_energy = dav.get_energy(i) + core_energy; // Ensure this function is implemented
            energies.push_back(total_energy);
            std::cout << "State " << i << ": Energy = " << total_energy << " Hartree\n";
        }
    }
    std::ofstream outFile("energies.txt");
    for (size_t i = 0; i < bond_lengths.size(); ++i)
    {
        outFile << "Bond length " << bond_lengths[i] << " Angstroms: ";
        for (int j = 0; j < 6; ++j)
        {
            outFile << energies[6 * i + j] << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
    // Plotting logic goes here
}