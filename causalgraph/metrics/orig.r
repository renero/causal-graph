allDagsJonas <- function (adj,row.names)
{
    a <- adj[row.names,row.names]
    
    if(any((a + t(a))==1))
    {
        #warning("The matrix is not entirely undirected.")
        return(-1)
    }
    return(allDagsIntern(adj, a, row.names,NULL))
}