function X = ParallelSP(m, n, k, phi, A, Y, spIterations, cglsIterations)
	X = spalloc(m, n, n * k);
	phi2 = parallel.pool.Constant(phi);
	A2 = parallel.pool.Constant(A);
	parfor j = 1:n
		rec = SP(k, phi2.Value, A2.Value, Y(:, j), spIterations, cglsIterations);
		X(:, j) = rec.x_hat;
	end
	clear phi2;
	clear A2;
end