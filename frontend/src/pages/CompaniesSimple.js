import React, { useState, useEffect } from 'react';
import api from '../services/api';

export default function CompaniesSimple() {
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchCompanies = async () => {
      try {
        const response = await api.get('/companies');
        setCompanies(response.data);
      } catch (error) {
        console.error('Error fetching companies:', error);
        // Données de démonstration si l'API ne fonctionne pas
        setCompanies([
          {
            id: 1,
            siren: '123456789',
            name: 'Entreprise Demo 1',
            address: '123 Rue de la Paix, Paris',
            status: 'prospect',
            chiffreAffaires: 1500000,
            effectifs: 25
          },
          {
            id: 2,
            siren: '987654321',
            name: 'Entreprise Demo 2',
            address: '456 Avenue des Champs, Lyon',
            status: 'contacté',
            chiffreAffaires: 2800000,
            effectifs: 45
          },
          {
            id: 3,
            siren: '456789123',
            name: 'Entreprise Demo 3',
            address: '789 Boulevard Victor Hugo, Marseille',
            status: 'qualifié',
            chiffreAffaires: 950000,
            effectifs: 15
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchCompanies();
  }, []);

  const filteredCompanies = companies.filter(company =>
    company.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    company.siren.includes(searchTerm)
  );

  const getStatusColor = (status) => {
    switch (status) {
      case 'prospect':
        return 'bg-yellow-100 text-yellow-800';
      case 'contacté':
        return 'bg-blue-100 text-blue-800';
      case 'qualifié':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="sm:flex sm:items-center">
        <div className="sm:flex-auto">
          <h1 className="text-2xl font-bold text-gray-900">Entreprises</h1>
          <p className="mt-2 text-sm text-gray-700">
            Liste des entreprises dans votre base de données
          </p>
        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
          <button
            type="button"
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 sm:w-auto"
          >
            Ajouter une entreprise
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="max-w-lg">
        <input
          type="text"
          placeholder="Rechercher par nom ou SIREN..."
          className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {/* Companies Table */}
      <div className="mt-8 flex flex-col">
        <div className="-my-2 -mx-4 overflow-x-auto sm:-mx-6 lg:-mx-8">
          <div className="inline-block min-w-full py-2 align-middle md:px-6 lg:px-8">
            <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
              <table className="min-w-full divide-y divide-gray-300">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Entreprise
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      SIREN
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Statut
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      CA
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Effectifs
                    </th>
                    <th scope="col" className="relative px-6 py-3">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredCompanies.map((company) => (
                    <tr key={company.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {company.name}
                          </div>
                          <div className="text-sm text-gray-500">
                            {company.address}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {company.siren}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(company.status)}`}>
                          {company.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {company.chiffreAffaires?.toLocaleString()}€
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {company.effectifs}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button className="text-blue-600 hover:text-blue-900 mr-4">
                          Modifier
                        </button>
                        <button className="text-red-600 hover:text-red-900">
                          Supprimer
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              {filteredCompanies.length === 0 && (
                <div className="text-center py-8">
                  <p className="text-gray-500">Aucune entreprise trouvée</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}