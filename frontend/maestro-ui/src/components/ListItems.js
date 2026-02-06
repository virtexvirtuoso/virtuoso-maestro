import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AssessmentIcon from '@mui/icons-material/Assessment';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';

export default function MainListItems() {
  const navigate = useNavigate();
  const location = useLocation();
  const currentPath = '/' + location.pathname.split('/')[1];

  const menuItems = [
    { path: '/', label: 'Home', icon: <DashboardIcon /> },
    { path: '/results', label: 'Results', icon: <AssessmentIcon /> },
    { path: '/evaluate', label: 'Evaluation', icon: <TrendingUpIcon /> },
    { path: '/compare', label: 'Compare', icon: <CompareArrowsIcon /> },
  ];

  return (
    <List>
      {menuItems.map((item) => (
        <ListItem key={item.path} disablePadding>
          <ListItemButton
            selected={currentPath === item.path}
            onClick={() => navigate(item.path)}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItemButton>
        </ListItem>
      ))}
    </List>
  );
}
