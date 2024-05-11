#pragma once
#include <string>
#include <vector>
#include <bitset>

/*** Written by Stephen J. Cotton ***/

template <size_t N>
class bitset: public std::bitset <N>
{
	using std::bitset <N>::bitset;

  public:
	bool is_set (int i) const { return (*this)[i]; }
};

inline
bool same_spin( int norbitals, int i, int a )
{
	const bool
		i_up = i<norbitals,
		a_up = a<norbitals;

	return i_up && a_up || !i_up && !a_up;
}

template <size_t N>
int count_intervening_ones (const bitset<N> &b, int l, int r)
{
	// Leave for students to implement
	throw "count_intervening_ones (...) requires implementation";
	return 0; 
}

template <size_t N>
bool is_neg (const bitset <N> &ket, int i, int a ) { return count_intervening_ones (ket,i,a)%2; }

template <size_t N>
bool is_neg (bitset <N> ket, int i, int j, int a, int b)
{
	const bool
		neg_ai = is_neg (ket, i,a),
		neg_bj = is_neg (ket.flip (i).flip (a), j,b);

	return neg_ai && !neg_bj || !neg_ai && neg_bj;
}

class integrals
{
	class integrals_impl *pimpl;

  public:
	explicit integrals (int n, std::string file);
	~integrals ();
	double get_core_energy () const;
	double get (int i, int a) const;
	double get (int i, int j, int a, int b) const;
	int norbitals () const;
	bool same_spin( int i, int a ) const { return ::same_spin (norbitals (), i,a); }
	double sget( int i, int j, int a, int b ) const
	{
		return get (i,j,a,b) - (same_spin (i,j)? get (j,i,a,b) :0);
	}

	template <size_t N>
	double calc_zero_move_matrix_element (const bitset <N> &ket) const
	{
		double H=0;
		for (int a=0; a<2*norbitals (); ++a) if (ket.is_set (a))
		{
			// 1-particle operator
			H += get (a,a);

			// 2-particle operator
			for (int b=0; b<a; ++b)
				if (ket.is_set (b))
					H += sget (a,b,a,b);
		}
		return H;
	}

	template <size_t N>
	double calc_one_move_matrix_element (const bitset <N> &ket, int i, int a) const
	{
		if (!same_spin (i,a)) return 0;

		// 1-particle operator
		double H = get (i,a);

		// 2-particle operator
		for (int b=0; b<2*norbitals (); ++b)
			if (ket.is_set (b))
				H += sget (i,b,a,b);

		if (H && is_neg (ket, i,a)) H *= -1;
		return H;
	}

	template <size_t N>
	double calc_two_move_matrix_element (const bitset <N> &ket, int i, int j, int a, int b) const
	{
		if (!same_spin (i,a) || !same_spin (j,b)) return 0;

		// Only 2-particle operator
		double H = sget (i,j,a,b);

		if (H && is_neg (ket, i,j,a,b)) H *= -1;
		return H;
	}
};
