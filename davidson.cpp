#pragma once
#include "your_headers.h++"

/*** Written by Stephen J. Cotton ***/

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

class Davidson
{
    std::vector<vec> basis, Hbasis; // vec is like valarray <double>; use the vector class of your choosing
    std::vector<double> saved_energies;
    mutable mat K;

public:
    size_t basis_size() const { return basis.size(); }
    size_t full_dim() const { return basis[0].size(); }
    double get_energy() const { return saved_energies.back(); }
    static bool check_tol(double a, double b) { return std::abs(a - b) < 1e-9; }
    bool has_stopped() const
    {
        // Do a double-check on the energies for extra safety
        const auto &x = saved_energies;
        const int n = x.size();
        return n >= 3 && check_tol(x[n - 1], x[n - 2]) && check_tol(x[n - 1], x[n - 3]);
    }

    explicit Davidson(size_t n)
    {
        // Initialize basis with normalized random vector
        vec x = gen_ran_vec(n);
        basis.emplace_back(x /= sqrt(inner_product(x, x)));
    }

    const mat &update_K() const // mat is a simple dense matrix class; use one of your choosing
    {
        const int
            nold = assert_equivalence(K.rows(), K.cols(), "K must be a square matrix"),
            n = assert_equivalence(basis.size(), Hbasis.size(), "update_K");

        // Incremental update to subspace matrix
        mat Kold = K;
        K.resize(n, n);

        // Copy old matrix elements
        for (int i = 0; i < nold; ++i)
            for (int j = 0; j < nold; ++j)
                K(i, j) = Kold(i, j);

        // Calculate (and save) new row and column
        for (int j = nold; j < n; ++j)
            for (int i = 0; i < j + 1; ++i)
                K(j, i) = K(i, j) = inner_product(basis[i], Hbasis[j]);

        return K;
    }

    template <typename op> // op is your sparse matrix which provides  a matrix-vector "dot" product
    void do_iter(const op &H)
    {
        // Check sizes: your op needs to provide the number of rows and cols it represents
        assert_equivalence(full_dim(), H.rows(), H.cols(), "do_iter (H)");

        // "dot" (ie multiply) H (Note that H is only applied ONCE per iteration and saved)
        Hbasis.emplace_back(H.dot(basis.back())); // op must implement "dot"

        // Solve subspace problem (after updating K)
        const auto [E, U] = sjc::lapack::solve_eigensystem(update_K()); // You have this from PS4
        saved_energies.push_back(E[0]);

        // Construct the diagonally-preconditioned residual vector
        const vec c = U.get_col(0);       // Your dense matrix must provide a column-vector getter
        const auto &D = H.get_diagonal(); // op must provide the diagonal D
        vec x(0, full_dim());             // Note arguments to constructor reversed versus std::vector
        for (int j = 0; j < c.size(); ++j)
        {
            const vec &Hb = Hbasis[j], &b = basis[j];

            for (int k = 0; k < full_dim(); ++k)
                x[k] += c[j] * (Hb[k] - E[0] * b[k]) / (E[0] - D[k]);
        }

        // Orthogonalize: you need to define an inner product operation on two vecs
        for (int j = 0; j < basis.size(); ++j)
            x -= inner_product(x, basis[j]) * basis[j];

        // Normalize and save (again, uses your inner_product)
        basis.push_back(x /= sqrt(inner_product(x, x)));
    }

    template <typename op>
    Davidson &build_for_ground_state(const op &H)
    {
        for (int iter = 0; iter < full_dim() && !has_stopped(); ++iter)
            do_iter(H);
        return *this;
    }
};

class DavidsonSEM : public Davidson
{
public:
    DavidsonSEM(size_t n) : Davidson(n) {}

    void do_iter_multi(const Eigen::SparseMatrix<double> &H)
    {
        // Extend the single iteration method to handle multiple eigenvectors
    }

    void build_for_multiple_states(const Eigen::SparseMatrix<double> &H, int num_states)
    {
        while (!has_stopped())
        {
            do_iter_multi(H);
        }
    }
};

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

    // Plotting logic goes here
}