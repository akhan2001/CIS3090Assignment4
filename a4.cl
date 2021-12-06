__kernel void a4(int size, __global char *above_row, __global char *curr_row) {

   uint nWorkers = get_local_size(0);
   uint my_rank = get_local_id(0);

   int start = (size / nWorkers) * my_rank;
   int end = (size / nWorkers) * (my_rank + 1);
   if (my_rank == (nWorkers - 1))
      end += (size % nWorkers);

   int shouldLive = 1;
   for (int i = start; i < end; i++)
   {
      int neighboursalive = 0;

      if (i - 2 >= 0)
         if (above_row[i-2] != ' ') neighboursalive++;
      if (i - 1 >= 0)
         if (above_row[i-1] != ' ') neighboursalive++;

      if (i + 1 <= size-1)
         if (above_row[i+1] != ' ') neighboursalive++;
      if (i + 2 <= size-1)
         if (above_row[i+2] != ' ') neighboursalive++;

      if (above_row[i] == ' ' && (neighboursalive == 2 || neighboursalive == 3) ){
				if(get_global_size(0) > 10){
					curr_row[i] = 'X';
				}
				else{
					curr_row[i] = (char)(my_rank + 48);
				}
	  	}
      else if (above_row[i] != ' ' && (neighboursalive == 2 || neighboursalive == 4) ){
				if(get_global_size(0) > 10){
					curr_row[i] = 'X';
				}
				else{
					curr_row[i] = (char)(my_rank + 48);
				}
	  	}
   }
}
